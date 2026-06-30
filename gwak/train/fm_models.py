import os
import wandb
import math
import time
import yaml
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from matplotlib.patches import Patch
from typing import Sequence, Optional
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from pytorch_lightning.callbacks import EarlyStopping
from einops import rearrange, repeat

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform

from gwak.train.losses import SupervisedSimCLRLoss
from gwak.train.schedulers import WarmupCosineAnnealingLR
from gwak.train.cl_models import Crayon


class GwakBaseModelClass(pl.LightningModule):

    def get_logger(self):
        logger_name = 'GwakBaseModelClass'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        return logger

class Standardizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean.to(x)) / self.std.to(x)

class FlowWrapper(nn.Module):
    def __init__(self, flow, standardizer=None, use_correlation=True,
                 condition_on_correlation=False, standardizer_x=None, standardizer_c=None):
        super().__init__()
        self.flow = flow
        self.standardizer = standardizer
        self.use_correlation = use_correlation
        self.condition_on_correlation = condition_on_correlation
        self.standardizer_x = standardizer_x
        self.standardizer_c = standardizer_c

    def forward(self, x, context):
        if self.condition_on_correlation:
            x_in = self.standardizer_x(x) if self.standardizer_x is not None else x
            c_in = self.standardizer_c(context) if self.standardizer_c is not None else context
            return self.flow.log_prob(inputs=x_in, context=c_in)
        elif self.use_correlation:
            x_full = torch.cat([x, context], dim=-1)
            if self.standardizer is not None: x_full = self.standardizer(x_full)
            return self.flow.log_prob(inputs=x_full)
        else:
            x_in = self.standardizer(x) if self.standardizer is not None else x
            return self.flow.log_prob(inputs=x_in)

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):

        torch.cuda.empty_cache()

        # Load best model
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path,
            **pl_module.hparams['init_args']
        )
        module.model.eval()

        # Wrap flow in a traceable nn.Module
        use_corr = getattr(module, 'use_correlation', True)
        use_cond = getattr(module, 'condition_on_correlation', False)
        wrapper = FlowWrapper(
            module.model,
            standardizer=module.standardizer if not use_cond else None,
            use_correlation=use_corr,
            condition_on_correlation=use_cond,
            standardizer_x=getattr(module, 'standardizer_x', None) if use_cond else None,
            standardizer_c=getattr(module, 'standardizer_c', None) if use_cond else None,
        ).to("cuda:0")
        wrapper.eval()
        example_input = torch.randn(1, module.embedding_dims).to("cuda:0")
        # if module.conditioning:
        example_context = torch.randn(1, 1).to("cuda:0")
        traced = torch.jit.trace(wrapper, (example_input, example_context))
        # traced = torch.jit.script(wrapper)

        # else:
        #     traced = torch.jit.trace(wrapper, (example_input))

        # Save the traced model
        save_dir = trainer.logger.log_dir or trainer.logger.save_dir
        os.makedirs(save_dir, exist_ok=True)
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
        X = torch.randn(1, 16).to(module.device) # GWAK 2
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
        self.fc11 = nn.Linear(hidden_dim, 16)  # Hidden layer
        self.fc12 = nn.Linear(16, 8)  # Hidden layer
        self.fc2 = nn.Linear(8, 1)  # Output layer
        self.activation = nn.ReLU()  # Non-linearity
        self.sigmaboy = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))  # Apply activation
        x = self.activation(self.fc11(x))  # Apply activation
        x = self.activation(self.fc12(x))  # Apply activation
        x = self.sigmaboy(self.fc2(x))  # Output score (no sigmoid)
        return x

class NonLinearClassifier(GwakBaseModelClass):
    def __init__(
            self,
            embedding_path: str=None,
            embedding_model: str=None,
            means: Optional[str] = None,
            stds: Optional[str] = None,
            new_shape=128,
            n_dims=16,
            c_path=None,
            conditioning=None,
            learning_rate=1e-3,
            hidden_dim=32):
        super().__init__()
        self.model = MLPModel(n_dims, hidden_dim)

        if embedding_model:
            self.graph = torch.jit.load(embedding_model)
            self.graph.eval()
            self.graph.to(device=self.device)
        else:
            self.graph = None

        self.new_shape = new_shape
        self.learning_rate = learning_rate

        # Store mean and std for standardization
        self.embedding_mean = torch.tensor(np.load(means)).to(self.device) if means is not None else None
        self.embedding_std = torch.tensor(np.load(stds)).to(self.device) if stds is not None else None
        self.standardizer = Standardizer(self.embedding_mean,self.embedding_std) if means is not None else None

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):

        batch, labels = batch
        unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
        # self.get_logger().info(f'Background: {unique[-1]}', dict(zip(unique, counts)))
        # 9 and 10 are background labels fixed from the dataset
        # Create masks
        background_mask = (labels == unique[-1]) | (labels == unique[-2])
        signal_mask = ~background_mask  # everything else

        signal_feats = batch[signal_mask]
        background_feats = batch[background_mask]

        # Feature extraction
        if self.graph is not None:
            signal_feats = self.graph(signal_feats)
            background_feats = self.graph(background_feats)

        if self.standardizer is not None:
            signal_feats = self.standardizer(signal_feats)
            background_feats = self.standardizer(background_feats)

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
        # self.get_logger().info(f"Signal loss {loss_signal}, background loss {loss_background}")

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        batch, labels = batch
        unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
        # self.get_logger().info(f'Background: {unique[-1]}', dict(zip(unique, counts)))
        # 9 and 10 are background labels fixed from the dataset
        # Create masks
        background_mask = (labels == unique[-1]) # | (labels == 10)
        signal_mask = ~background_mask  # everything else

        signal_feats = batch[signal_mask]
        background_feats = batch[background_mask]

        # Feature extraction
        if self.graph is not None:
            signal_feats = self.graph(signal_feats)
            background_feats = self.graph(background_feats)

        if self.standardizer is not None:
            signal_feats = self.standardizer(signal_feats)
            background_feats = self.standardizer(background_feats)

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


class BackgroundFlowModel(GwakBaseModelClass):
    def __init__(
            self,
            embedding_model: str = None,
            new_shape=128,
            n_dims=8,
            n_flow_steps=4,
            hidden_dim=64,
            use_correlation: bool = True,
            condition_on_correlation: bool = False,
            normalize: bool = True,
            learning_rate=1e-3,
    ):
        super().__init__()

        # Load feature extractor
        self.embedding_model = embedding_model
        if self.embedding_model:
            self.graph = torch.jit.load(self.embedding_model)
            self.graph.eval()
            self.graph.to(device=self.device)
            self.get_logger().info(f"Loaded embedding model from {embedding_model}")

        self.use_correlation = use_correlation
        self.condition_on_correlation = condition_on_correlation
        self.normalize = normalize
        self.n_dims = n_dims
        # embedding_dims: pure embedding size (no xcorr)
        if condition_on_correlation:
            self.embedding_dims = n_dims          # flow input = embeddings only
            context_features = 1
        elif use_correlation:
            self.embedding_dims = n_dims - 1      # flow input = embedding + xcorr concatenated
            context_features = None
        else:
            self.embedding_dims = n_dims
            context_features = None

        # Define RQS Flow
        transforms = []
        for _ in range(n_flow_steps):
            maf = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.n_dims,
                hidden_features=hidden_dim,
                num_blocks=4,
                num_bins=8,
                tail_bound=5,
                tails='linear',
                context_features=context_features,
            )
            transforms.append(maf)
            transforms.append(ReversePermutation(features=self.n_dims))

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([self.n_dims])
        )

        if condition_on_correlation:
            # Separate standardizers for embeddings and context
            self.standardizer = None
            self.standardizer_x = Standardizer(torch.zeros(n_dims), torch.ones(n_dims))
            self.standardizer_c = Standardizer(torch.zeros(1), torch.ones(1))
        else:
            self.standardizer = Standardizer(torch.zeros(n_dims), torch.ones(n_dims))
            self.standardizer_x = None
            self.standardizer_c = None
        self.model = self.flow

        self.new_shape = new_shape
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def frequency_cos_similarity(self, batch):
        H = torch.fft.rfft(batch[:, 0, :], dim=-1)
        L = torch.fft.rfft(batch[:, 1, :], dim=-1)
        numerator = torch.sum(H * torch.conj(L), dim=-1)
        norm_H = torch.linalg.norm(H, dim=-1)
        norm_L = torch.linalg.norm(L, dim=-1)
        rho_complex = numerator / (norm_H * norm_L + 1e-8)
        rho_real = torch.real(rho_complex).unsqueeze(-1)
        return rho_real

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def on_fit_start(self):
        if not self.normalize:
            return
        dm = self.trainer.datamodule
        if not hasattr(dm, 'x_mean'):
            return
        if self.condition_on_correlation:
            self.standardizer_x.mean.copy_(dm.x_mean)
            self.standardizer_x.std.copy_(dm.x_std)
            if hasattr(dm, 'c_mean'):
                self.standardizer_c.mean.copy_(dm.c_mean)
                self.standardizer_c.std.copy_(dm.c_std)
        elif self.use_correlation and hasattr(dm, 'c_mean'):
            mean = torch.cat([dm.x_mean, dm.c_mean], dim=0)
            std = torch.cat([dm.x_std, dm.c_std], dim=0)
            self.standardizer.mean.copy_(mean)
            self.standardizer.std.copy_(std)
        else:
            self.standardizer.mean.copy_(dm.x_mean)
            self.standardizer.std.copy_(dm.x_std)

    def _get_feats_and_c(self, batch):
        need_c = self.use_correlation or self.condition_on_correlation
        if self.embedding_model:
            if len(batch) == 2:
                batch, _ = batch
            feats = self.graph(batch)
            c = self.frequency_cos_similarity(batch) if need_c else None
        else:
            feats = batch[0]
            c = batch[1] if need_c and len(batch) > 1 else None
        return feats, c

    def training_step(self, batch, batch_idx):
        feats, c = self._get_feats_and_c(batch)

        if self.condition_on_correlation:
            feats_in = self.standardizer_x(feats)
            c_in = self.standardizer_c(c)
            log_prob = self.model.log_prob(inputs=feats_in, context=c_in)
        elif self.use_correlation:
            feats_in = self.standardizer(torch.cat([feats, c], dim=-1))
            log_prob = self.model.log_prob(inputs=feats_in)
        else:
            feats_in = self.standardizer(feats)
            log_prob = self.model.log_prob(inputs=feats_in)

        loss = -log_prob.mean()
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        feats, c = self._get_feats_and_c(batch)

        if self.condition_on_correlation:
            feats_in = self.standardizer_x(feats)
            c_in = self.standardizer_c(c)
            log_prob = self.model.log_prob(inputs=feats_in, context=c_in)
        elif self.use_correlation:
            feats_in = self.standardizer(torch.cat([feats, c], dim=-1))
            log_prob = self.model.log_prob(inputs=feats_in)
        else:
            feats_in = self.standardizer(feats)
            log_prob = self.model.log_prob(inputs=feats_in)

        loss = -log_prob.mean()
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        return [
            ModelCheckpoint(
                monitor='val/loss',
                save_last=True,
                auto_insert_metric_name=False
            ),
            pl.callbacks.EarlyStopping(
                monitor='val/loss',
                patience=10,
                mode='min',
                verbose=True
            )
        ]
