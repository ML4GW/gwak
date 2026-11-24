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
from pathlib import Path
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
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform, 
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
)

from train.losses import SupervisedSimCLRLoss
from train.schedulers import WarmupCosineAnnealingLR
from train.cl_models import Crayon
from transforms import cohmode_to_dim
from transforms.preprocess import FeatureExtractor

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
    def __init__(
        self, 
        flow, 
        input_standardizer=None,
        context_standardizer=None,
        # concate_context=True,
        # condition_on_correlation=False, 
        # standardizer_x=None, 
        # standardizer_c=None
    ):
        super().__init__()
        self.flow = flow
        self.input_standardizer = input_standardizer
        self.context_standardizer =  context_standardizer

    def forward(
        self,
        x,
        context:Optional[torch.Tensor]=None
    ):
        if self.input_standardizer:
            x = self.input_standardizer(x)
        if self.context_standardizer:
            context = self.context_standardizer(context)
            return self.flow.log_prob(inputs=x, context=context)
        return self.flow.log_prob(inputs=x)

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):

        torch.cuda.empty_cache()

        # Load best model
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path,
            **pl_module.hparams
        )
        module.model.eval()

        # Wrap flow in a traceable nn.Module
        _device = "cuda:0"
        use_corr = getattr(module, "concate_context", True)
        # use_cont = getattr(module, 'apply_context', False)
        wrapper = FlowWrapper(
            module.model,
            input_standardizer=module.input_standardizer,
            context_standardizer=module.context_standardizer
        ).to(_device)
        wrapper.eval()

        # this is for spline
        # example_input = torch.randn(1, module.embedding_dims).to("cuda:0")
        example_input = torch.randn(1, module.input_dim).to(_device)
        if module.apply_context:
            example_context = torch.randn(1, module.context_dim).to(_device)
            traced = torch.jit.trace(wrapper, (example_input, example_context))
        else:
            traced = torch.jit.trace(wrapper, (example_input))

        # Save the traced model
        save_dir = trainer.logger.log_dir or trainer.logger.save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "model_JIT.pt"
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
        small_bkgs = self.backgrounds[
            i * self.new_shape:(i + 1) * self.new_shape
        ].to(self.device)

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
            condition_on_correlation=None,
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
            embedding_dim=8,
            n_flow_steps=4, #Bump this up to ~20
            num_blocks=4,
            hidden_dim=64,
            feats_mode: str="embedding",
            feature_mode:Optional[str]=None,
            # num_bins=50,
            learning_rate=1e-3,
            concate_context: bool = True,
            apply_context: bool = False,
            normalize: bool = True,
    ):
        super().__init__()

        # Load feature extractor
        self.embedding_model = embedding_model
        if self.embedding_model:
            self.graph = torch.jit.load(self.embedding_model)
            self.graph.eval()
            self.graph.to(device=self.device)
            self.get_logger().info(
                f"Loaded embedding model from {embedding_model}"
            )

        self.embedding_dim = embedding_dim
        self.concate_context = concate_context
        self.apply_context = apply_context
        self.normalize = normalize
        self.context_dim = cohmode_to_dim(feature_mode)
        self.feats_mode = feats_mode

        if feats_mode == "embedding":
            self.input_dim = embedding_dim
            self.context_dim = cohmode_to_dim(feature_mode)
        else:
            self.input_dim = cohmode_to_dim(feature_mode)
            self.context_dim = embedding_dim

        if concate_context:
            # flow input = embedding + context concatenated 
            self.input_dim += self.context_dim
            self.context_dim = 0
        if apply_context == False:
            self.context_dim = 0


        # Define MAF Flow
        transforms = []
        for _ in range(n_flow_steps):
            maf = MaskedAffineAutoregressiveTransform(
                features=self.input_dim,
                hidden_features=hidden_dim,
                num_blocks=num_blocks, # Add this to config arguments ~32
                context_features=self.context_dim
            )
            transforms.append(maf)
            transforms.append(ReversePermutation(features=self.input_dim))

        # if not self.apply_context: 
        #     distribution = StandardNormal([self.input_dim])
        # else:
        #     context_encoder=nn.Linear(self.context_dim, 2*self.input_dim)
        #     distribution = ConditionalDiagonalNormal(
        #         [self.input_dim],
        #         context_encoder=context_encoder
        #     )
        distribution = StandardNormal([self.input_dim])

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=distribution
        )
        # Store mean and std for standardization
        self.input_standardizer = Standardizer(
            torch.zeros(self.input_dim),
            torch.ones(self.input_dim)
        )
        self.context_standardizer = None
        if apply_context:
            self.context_standardizer = Standardizer(
                torch.zeros(self.context_dim),
                torch.ones(self.context_dim)
            )

        self.model = self.flow
        self.learning_rate = learning_rate

        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor(
            mode=feature_mode
        )
    def setup(self, stage=None):
        if stage != "fit" or self.context_standardizer is None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        features_mean = getattr(datamodule, "features_mean", None)
        features_std = getattr(datamodule, "features_std", None)
        embd_mean = getattr(datamodule, "embd_mean", None)
        embd_std = getattr(datamodule, "embd_std", None)
        if features_mean is None or features_mean.numel() != self.context_dim:
            self.get_logger().warning(
                "No context statistics from the datamodule, context is not standardized"
            )
            return

        self.embd_standardizer.mean.copy_(embd_mean)
        self.embd_standardizer.std.copy_(embd_std)
        self.context_standardizer.mean.copy_(features_mean)
        self.context_standardizer.std.copy_(features_std)

    def freq_cos_sim(self, batch):
        sim_score = self.feature_extractor(batch)
        return sim_score

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        if self.embedding_model:
            if len(batch) == 2:
                batch, _ = batch
            inputs = self.graph(batch)
            context = self.freq_cos_sim(batch)
        else:
            inputs, context = batch

        inputs = self.input_standardizer(inputs)

        if self.apply_context:
            context = self.context_standardizer(context)
            log_prob = self.model.log_prob(inputs=inputs, context=context)
        else:
            log_prob = self.model.log_prob(inputs=inputs)

        loss = -log_prob.mean()
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.embedding_model:
            if len(batch) == 2:
                batch, _ = batch
            inputs = self.graph(batch)
            context = self.freq_cos_sim(batch)
        else:
            inputs, context = batch

        inputs = self.input_standardizer(inputs)
        if self.apply_context:
            context = self.context_standardizer(context)
            log_prob = self.model.log_prob(inputs=inputs, context=context)
        else:
            log_prob = self.model.log_prob(inputs=inputs)

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