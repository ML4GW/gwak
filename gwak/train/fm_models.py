import os
import time
import yaml
import logging
import h5py
from typing import Sequence
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning.pytorch as pl

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from losses import SupervisedSimCLRLoss
from schedulers import WarmupCosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import wandb
from PIL import Image
from io import BytesIO

from cl_models import Crayon

class GwakBaseModelClass(pl.LightningModule):

    def get_logger(self):
        logger_name = 'GwakBaseModelClass'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        return logger


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path,
            **pl_module.hparams['init_args']
        )

        module.model.eval()
        # Modifiy these to read the last training/valdation data
        # to acess the input shape.
        # X = torch.randn(1, 200, 2) # GWAK 1
        X = torch.randn(1, 2, 200) # GWAK 2

        # trace = torch.jit.trace(module.model.to("cpu"), X.to("cpu"))
        trace = torch.jit.script(module.model.to("cpu"))
        save_dir = trainer.logger.log_dir or trainer.logger.save_dir

        with open(os.path.join(save_dir, "model_JIT.pt"), "wb") as f:
            torch.jit.save(trace, f)


import torch.nn as nn

class FlowWrapper(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x):
        return self.flow.log_prob(x)

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):
        import torch.nn as nn
        import os

        torch.cuda.empty_cache()

        # Load best model
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path,
            **pl_module.hparams['init_args']
        )
        module.model.eval()

        # Wrap flow in a traceable nn.Module
        wrapper = FlowWrapper(module.model).cpu()
        wrapper.eval()

        # Dummy input: must match the expected shape of flow input
        # For example, if input to flow is (batch_size, 8):
        example_input = torch.randn(1, module.model._transform._transforms[0].features)

        # Trace the wrapped model
        traced = torch.jit.trace(wrapper, example_input)

        # Save the traced model
        save_dir = trainer.logger.log_dir or trainer.logger.save_dir
        save_path = os.path.join(save_dir, "model_JIT.pt")
        traced.save(save_path)


class LinearModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path,
            **pl_module.hparams['init_args']
        )

        # Modifiy these to read the last training/valdation data
        # to acess the input shape.
        X = torch.randn(1, 8).to(module.device) # GWAK 2
        trace = torch.jit.trace(module.model.to("cpu"), X.to("cpu"))

        save_dir = trainer.logger.log_dir or trainer.logger.save_dir

        with open(os.path.join(save_dir, "mlp_model_JIT.pt"), "wb") as f:
            torch.jit.save(trace, f)


class Linear(GwakBaseModelClass):
    def __init__(
            self,
            backgrounds: str = '/home/hongyin.chen/whiten_timeslide.h5', # timeslides to train against
            ckpt: str = "../output/test_S4_fixedSignals_0p5sec_v2/lightning_logs/2wla29uz/checkpoints/33-1700.ckpt",
            cfg_path: str = "../output/test_S4_fixedSignals_0p5sec_v2/config.yaml", # pre-trained embedding model
            new_shape=128,
            n_dims=16,
            learning_rate=1e-3):
        super().__init__()
        self.model = nn.Linear(n_dims, 1)

        with h5py.File(backgrounds, "r") as h:
            # Load data as a NumPy array first
            data = h["data"][:128*10*20, :, :1024]  # Ensuring it's a NumPy array

        # Splitting into train and validation sets (80% train, 20% validation)
        self.backgrounds, self.val_backgrounds = train_test_split(data, test_size=0.2, random_state=42)
        self.backgrounds = torch.from_numpy(self.backgrounds).to("cpu")
        self.val_backgrounds = torch.from_numpy(self.val_backgrounds).to("cpu")

        self.ckpt = ckpt
        self.cfg_path = cfg_path
        # Load first model (frozen for inference)
        with open(self.cfg_path,"r") as fin:
            cfg = yaml.load(fin,yaml.FullLoader)
        self.graph = Crayon.load_from_checkpoint(self.ckpt, **cfg['model']['init_args'])
        self.graph.eval()
        self.graph.to(device=self.device)

        self.new_shape = new_shape
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        i = batch_idx  # Each batch corresponds to a background chunk
        small_bkgs = self.backgrounds[i * self.new_shape:(i + 1) * self.new_shape].to(self.device)

        # Load first model (frozen for inference)
        batch = self.graph.model(batch[0])
        small_bkgs = self.graph.model(small_bkgs)

        background_MV = self.model(small_bkgs)
        signal_MV = self.model(batch)

        # Take min score for each signal example
        signal_MV = torch.min(signal_MV, dim=1)[0]

        zero = torch.tensor(0.0, device=self.device)

        background_loss = torch.maximum(zero, 1 - background_MV).mean()
        signal_loss = torch.maximum(zero, 1 + signal_MV).mean()

        loss = background_loss + signal_loss

        self.log(
            'train_loss',
            loss,
            on_epoch=True, # Newly added
            sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Validation step processes all validation data at once."""
        small_bkgs = self.val_backgrounds.to(self.device)

        # Load first model (frozen for inference)
        batch = self.graph.model(batch[0])
        small_bkgs = self.graph.model(small_bkgs)

        background_MV = self.model(small_bkgs)
        signal_MV = self.model(batch)

        # Take min score for each signal example
        signal_MV = torch.min(signal_MV, dim=1)[0]

        zero = torch.tensor(0.0, device=self.device)

        background_loss = torch.maximum(zero, 1 - background_MV).mean()
        signal_loss = torch.maximum(zero, 1 + signal_MV).mean()

        loss = background_loss + signal_loss
        self.log(
            'val/loss',
            loss,
            sync_dist=True)

        return loss

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        # checkpoint for saving best model
        # that will be used for downstream export
        # and inference tasks
        # checkpoint for saving multiple best models
        callbacks = []

        # if using ray tune don't append lightning
        # model checkpoint since we'll be using ray's
        checkpoint = LinearModelCheckpoint(
            monitor='val/loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer
        self.activation = nn.ReLU()  # Non-linearity
        self.sigmaboy = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))  # Apply activation
        x = self.sigmaboy(self.fc2(x))  # Output score (no sigmoid)
        return x

class NonLinearClassifier(GwakBaseModelClass):
    def __init__(
            self,
            backgrounds: str = '/home/hongyin.chen/whiten_timeslide.h5',
            ckpt: str = "../output/test_S4_fixedSignals_0p5sec_v2/lightning_logs/2wla29uz/checkpoints/33-1700.ckpt",
            cfg_path: str = "../output/test_S4_fixedSignals_0p5sec_v2/config.yaml",
            new_shape=128,
            n_dims=16,
            learning_rate=1e-3,
            hidden_dim=32):
        super().__init__()
        self.model = MLPModel(n_dims, hidden_dim)

        with h5py.File(backgrounds, "r") as h:
            data = h["data"][:128*10*20, :, :1024]

        self.backgrounds, self.val_backgrounds = train_test_split(data, test_size=0.2, random_state=42)
        self.backgrounds = torch.tensor(self.backgrounds, dtype=torch.float32).to("cpu")
        self.val_backgrounds = torch.tensor(self.val_backgrounds, dtype=torch.float32).to("cpu")

        self.ckpt = ckpt
        self.cfg_path = cfg_path

        with open(self.cfg_path, "r") as fin:
            cfg = yaml.load(fin, yaml.FullLoader)
        self.graph = Crayon.load_from_checkpoint(self.ckpt, **cfg['model']['init_args'])
        self.graph.eval()
        self.graph.to(device=self.device)

        self.new_shape = new_shape
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        i = batch_idx
        small_bkgs = self.backgrounds[i * self.new_shape:(i + 1) * self.new_shape].to(self.device)

        # Feature extraction
        signal_feats = self.graph.model(batch[0])
        background_feats = self.graph.model(small_bkgs)

        # Predictions
        signal_preds = self.forward(signal_feats)
        background_preds = self.forward(background_feats)

        # BCE targets: 1 = signal, 0 = background
        target_signal = torch.ones_like(signal_preds)
        target_background = torch.zeros_like(background_preds)

        # Compute BCE loss
        loss_signal = F.binary_cross_entropy(signal_preds, target_signal)
        loss_background = F.binary_cross_entropy(background_preds, target_background)
        loss = loss_signal + loss_background

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        small_bkgs = self.val_backgrounds.to(self.device)

        signal_feats = self.graph.model(batch[0])
        background_feats = self.graph.model(small_bkgs)

        signal_preds = self.forward(signal_feats)
        background_preds = self.forward(background_feats)

        target_signal = torch.ones_like(signal_preds)
        target_background = torch.zeros_like(background_preds)

        loss_signal = F.binary_cross_entropy(signal_preds, target_signal)
        loss_background = F.binary_cross_entropy(background_preds, target_background)
        loss = loss_signal + loss_background

        self.log("val/loss", loss, sync_dist=True)

        return loss

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        return [LinearModelCheckpoint(
            monitor='val/loss',
            save_last=True,
            auto_insert_metric_name=False
        )]


from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
import torch.nn as nn
class BackgroundFlowModel(GwakBaseModelClass):
    def __init__(
            self,
            ckpt: str = "output/S4_SimCLR_multiSignalAndBkg/lightning_logs/8wuhxd59/checkpoints/47-2400.ckpt",
            cfg_path: str = "output/S4_SimCLR_multiSignalAndBkg/config.yaml",
            new_shape=128,
            n_dims=8,
            n_flow_steps=4,
            hidden_dim=64,
            learning_rate=1e-3
    ):
        super().__init__()

        # Load feature extractor
        with open(cfg_path, "r") as fin:
            cfg = yaml.load(fin, yaml.FullLoader)
        self.graph = Crayon.load_from_checkpoint(ckpt, **cfg['model']['init_args'])
        self.graph.eval()
        self.graph.to(device=self.device)

        # MAF Flow (instead of RealNVP)
        transforms = []
        for _ in range(n_flow_steps):
            maf = MaskedAffineAutoregressiveTransform(
                features=n_dims,
                hidden_features=hidden_dim,
                num_blocks=2,
                use_residual_blocks=True,
                activation=nn.ReLU()
            )
            transforms.append(maf)
            transforms.append(ReversePermutation(features=n_dims))  # adds variety

        self.model = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([n_dims])
        )

        self.new_shape = new_shape
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def forward(self, x):
        return self.model.log_prob(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        feats = self.graph.model(batch)
        log_prob = self.model.log_prob(feats)
        loss = -log_prob.mean()
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        feats = self.graph.model(batch)
        log_prob = self.model.log_prob(feats)
        loss = -log_prob.mean()
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        return [ModelCheckpoint(
            monitor='val/loss',
            save_last=True,
            auto_insert_metric_name=False
        )]