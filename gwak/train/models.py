import os
import time
import yaml
import logging
import h5py
from typing import Sequence
from collections import OrderedDict
#from sklearn.model_selection import train_test_split

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

        # Modifiy these to read the last training/valdation data
        # to acess the input shape.
        # X = torch.randn(1, 200, 2) # GWAK 1
        X = torch.randn(1, 2, 200) # GWAK 2

        trace = torch.jit.trace(module.model.to("cpu"), X.to("cpu"))

        save_dir = trainer.logger.log_dir or trainer.logger.save_dir

        with open(os.path.join(save_dir, "model_JIT.pt"), "wb") as f:
            torch.jit.save(trace, f)


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
            ckpt: str = "../output/test_S4_fixedSignals_0p5sec_v2/lightning_logs/2wla29uz/checkpoints/27-1400.ckpt",
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

    def forward(self, x):
        x = self.activation(self.fc1(x))  # Apply activation
        x = self.fc2(x)  # Output score (no sigmoid)
        return x

class NonLinearClassifier(GwakBaseModelClass):
    def __init__(
            self,
            backgrounds: str = '/home/hongyin.chen/whiten_timeslide.h5', # timeslides to train against
            ckpt: str = "../output/test_S4_fixedSignals_0p5sec_v2/lightning_logs/2wla29uz/checkpoints/27-1400.ckpt",
            cfg_path: str = "../output/test_S4_fixedSignals_0p5sec_v2/config.yaml", # pre-trained embedding model
            new_shape=128,
            n_dims=16,
            learning_rate=1e-3,
            hidden_dim=32):
        super().__init__()
        self.model = MLPModel(n_dims, hidden_dim)

        with h5py.File(backgrounds, "r") as h:
            data = h["data"][:128*10*20, :, :1024]

        # Splitting into train and validation sets (80% train, 20% validation)
        self.backgrounds, self.val_backgrounds = train_test_split(data, test_size=0.2, random_state=42)
        self.backgrounds = torch.tensor(self.backgrounds, dtype=torch.float32).to("cpu")
        self.val_backgrounds = torch.tensor(self.val_backgrounds, dtype=torch.float32).to("cpu")

        self.ckpt = ckpt
        self.cfg_path = cfg_path
        # Load first model (frozen for inference)
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

        # Extract features from frozen model
        batch = self.graph.model(batch[0])
        small_bkgs = self.graph.model(small_bkgs)

        # Pass through MLP model
        background_MV = self.model(small_bkgs)
        signal_MV = self.model(batch)

        # Define target labels (+1 for background, -1 for signal)
        target_background = torch.ones_like(background_MV)
        target_signal = -torch.ones_like(signal_MV)

        # Mean Squared Error loss (forces background→+1, signal→-1)
        loss = F.mse_loss(background_MV, target_background) + F.mse_loss(signal_MV, target_signal)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        small_bkgs = self.val_backgrounds.to(self.device)

        # Extract features from frozen model
        batch = self.graph.model(batch[0])
        small_bkgs = self.graph.model(small_bkgs)

        # Pass through MLP model
        background_MV = self.model(small_bkgs)
        signal_MV = self.model(batch)

        # Define target labels (+1 for background, -1 for signal)
        target_background = torch.ones_like(background_MV)
        target_signal = -torch.ones_like(signal_MV)

        # Compute validation loss
        loss = F.mse_loss(background_MV, target_background) + F.mse_loss(signal_MV, target_signal)

        self.log("val/loss", loss, sync_dist=True)

        return loss

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        callbacks = []
        checkpoint = LinearModelCheckpoint(
            monitor='val/loss',
            save_last=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint)
        return callbacks

class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim // 2
        self.rnn1_0 = nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=2,
            batch_first=True
        )
        self.rnn1_1 = nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=2,
            batch_first=True
        )

        self.encoder_dense_scale = 20
        self.linear1 = nn.Linear(
            in_features=2**8,
            out_features=self.encoder_dense_scale * 4
        )
        self.linear2 = nn.Linear(
            in_features=self.encoder_dense_scale * 4,
            out_features=self.encoder_dense_scale * 2
        )
        self.linear_passthrough = nn.Linear(
            2 * seq_len,
            self.encoder_dense_scale * 2
        )
        self.linear3 = nn.Linear(
            in_features=self.encoder_dense_scale * 4,
            out_features=self.embedding_dim
        )

        self.linearH = nn.Linear(4 * seq_len, 2**7)
        self.linearL = nn.Linear(4 * seq_len, 2**7)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, 2 * self.seq_len)
        other_dat = self.linear_passthrough(x_flat)
        Hx, Lx = x[:, :, 0][:, :, None], x[:, :, 1][:, :, None]

        Hx, (_, _) = self.rnn1_0(Hx)
        Hx = Hx.reshape(batch_size, 4 * self.seq_len)
        Hx = F.tanh(self.linearH(Hx))

        Lx, (_, _) = self.rnn1_1(Lx)
        Lx = Lx.reshape(batch_size, 4 * self.seq_len)
        Lx = F.tanh(self.linearL(Lx))

        x = torch.cat([Hx, Lx], dim=1)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = torch.cat([x, other_dat], dim=1)
        x = F.tanh(self.linear3(x))

        return x.reshape((batch_size, self.embedding_dim))  # phil harris way

class Decoder(nn.Module):

    def __init__(self, seq_len, n_features=1, input_dim=64,):
        super().__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.rnn1_0 = nn.LSTM(
            input_size=1,
            hidden_size=1,
            num_layers=1,
            batch_first=True
        )
        self.rnn1_1 = nn.LSTM(
            input_size=1,
            hidden_size=1,
            num_layers=1,
            batch_first=True
        )
        self.rnn1 = nn.LSTM(
            input_size=2,
            hidden_size=2,
            num_layers=1,
            batch_first=True
        )

        self.linearH = nn.Linear(2 * self.seq_len, self.seq_len)
        self.linearL = nn.Linear(2 * self.seq_len, self.seq_len)

        self.linear1 = nn.Linear(self.hidden_dim, 2**8)
        self.linear2 = nn.Linear(2**8, 2 * self.seq_len)

    def forward(self, x):

        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        Hx = self.linearH(x)[:, :, None]
        Lx = self.linearL(x)[:, :, None]

        x = torch.cat([Hx, Lx], dim=2)

        return x

class LargeLinear(GwakBaseModelClass):

    def __init__(
            self,
            num_ifos=2,
            num_timesteps=200,
            bottleneck=8
        ):

        super(LargeLinear, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos

        self.model = nn.Sequential(OrderedDict([
            ("Reshape_Layer", nn.Flatten(1)), # Consider use torch.view() instead
            ("E_Linear1", nn.Linear(num_timesteps * 2, 2**7)),
            ("E_ReLU1", nn.ReLU()),
            ("E_Linear2", nn.Linear(2**7, 2**9)),
            ("E_ReLU2", nn.ReLU()),
            ("E_Linear3", nn.Linear(2**9, bottleneck)),
            ("E_ReLU3", nn.ReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ("D_Linear1", nn.Linear(bottleneck, 2**9)),
            ("D_ReLU1", nn.ReLU()),
            ("D_Linear2", nn.Linear(2**9, 2**7)),
            ("D_Tanh", nn.Tanh()),
            ("D_Linear3", nn.Linear(2**7, num_timesteps * 2)),
        ]))

    def training_step(self, batch, batch_idx):

        x = batch
        batch_size = x.shape[0]

        x = self.model(x)
        x = self.decoder(x)
        x = x.reshape(batch_size, self.num_ifos, self.num_timesteps)

        loss_fn = torch.nn.L1Loss()

        loss = loss_fn(batch, x)

        self.log(
            'train_loss',
            loss,
            on_epoch=True, # Newly added
            sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        x = batch
        batch_size = x.shape[0]

        x = x.reshape(-1, self.num_timesteps * self.num_ifos)
        x = self.model(x)
        x = self.decoder(x)
        x = x.reshape(batch_size, self.num_ifos, self.num_timesteps)

        loss_fn = torch.nn.L1Loss()

        self.metric = loss_fn(batch, x)

        self.log(
            'val_loss',
            self.metric,
            on_epoch=True,
            sync_dist=True
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        # checkpoint for saving best model
        # that will be used for downstream export
        # and inference tasks
        # checkpoint for saving multiple best models
        callbacks = []

        # if using ray tune don't append lightning
        # model checkpoint since we'll be using ray's
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks

class Autoencoder(GwakBaseModelClass):

    def __init__(
        self,
        num_ifos: int = 2,
        num_timesteps: int = 200,
        bottleneck: int = 8
        ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.bottleneck = bottleneck
        self.model = Encoder(
            seq_len=num_timesteps,
            n_features=num_ifos,
            embedding_dim=bottleneck
        )
        self.decoder = Decoder(
            seq_len=num_timesteps,
            n_features=num_ifos,
            input_dim=bottleneck
        )
        self.model___ = S4Model(d_input=self.num_ifos,
                    length=self.num_timesteps,
                    d_output = 10)

    def training_step(self, batch, batch_idx):
        #print(271, batch.shape)
        #assert 0
        x = batch

        x = x.transpose(1, 2)
        x = self.model(x)
        x = self.decoder(x)

        x = x.transpose(1, 2)

        loss_fn = torch.nn.L1Loss()

        self.metric = loss_fn(batch, x)

        self.log(
            'train_loss',
            self.metric,
            on_epoch=True,
            sync_dist=True
            )

        return self.metric

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch

        x = x.transpose(1, 2)
        x = self.model(x)
        x = self.decoder(x)

        x = x.transpose(1, 2)

        loss_fn = torch.nn.L1Loss()

        loss = loss_fn(batch, x)

        self.log(
            'val_loss',
            loss,
            on_epoch=True,
            sync_dist=True
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        # checkpoint for saving best model
        # that will be used for downstream export
        # and inference tasks
        # checkpoint for saving multiple best models
        callbacks = []

        # if using ray tune don't append lightning
        # model checkpoint since we'll be using ray's
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks


class gwak2(GwakBaseModelClass):

    def __init__(self, ):

        super().__init__()

    def training_step(self, batch, batch_idx):
        self.log(
            'train_loss',
            loss,
            sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        return optimizer


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), "
                "but got {}".format(p)
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            mask_shape = (
                X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            )
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(
        self,
        d_model: int,
        length: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = None,
    ):
        super().__init__()

        # generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        Ls = torch.arange(length)
        self.register_buffer("length", Ls)

    def forward(self):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * self.length  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """
        Register a tensor with a configurable learning rate
        and 0 weight decay
        """

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(
        self,
        d_model: int,
        length: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None,
    ):
        super().__init__()
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(d_model))
        self.length = length

        # SSM Kernel
        self.kernel = S4DKernel(
            d_model,
            length=length,
            N=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
            lr=lr,
        )

        # Pointwise
        self.activation = nn.GELU()
        # TODO: investigate torch dropout implementation
        self.dropout = torch.nn.Dropout1d(dropout)
        # self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)

        # Compute SSM Kernel
        k = self.kernel()  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * self.length)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * self.length)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * self.length)[
            ..., : self.length
        ]  # (B H L)

        # Compute D term in state space equation
        # Essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        # Return a dummy state to satisfy this repo's interface,
        # but this can be modified
        return y, None


class S4Model(nn.Module):
    def __init__(
        self,
        d_input: int,
        length: int,
        d_output: int = 10,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        prenorm: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        if lr is not None:
            lr = min(0.001, lr)
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    length=length,
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=True,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    lr=lr,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """
        x = x.transpose(-1, -2)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts
        ):
            # Each iteration of this loop will map
            # (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

class MLP(nn.Module):
    def __init__(self, d_input:int, hidden_dims:list[int], d_output, dropout=0.0, activation=nn.ReLU(), output_activation=None):
        super().__init__()
        #copying the paper of having one-layer MLP
        layers = []
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(d_input, d_output))
        else:
            dcurr = d_input
            for dh in hidden_dims:
                layers.append(nn.Linear(dcurr,dh))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
                dcurr = dh
            layers.append(nn.Linear(dcurr,d_output))
        if output_activation is not None:
            layers.append(output_activation)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Crayon(GwakBaseModelClass):

    def __init__(
        self,
        num_ifos: int = 2,
        num_timesteps: int = 200,
        d_output:int = 10,
        d_contrastive_space: int = 20,
        temperature: float = 0.1,
        supervised_simclr: bool = False,
        lr_opt: float = 1e-4,
        s4_kwargs: Optional[dict] = {}
        ):

        super().__init__()
        self.num_ifos = num_ifos
        self.num_timesteps = num_timesteps
        self.d_output = d_output
        self.d_contrastive_space = d_contrastive_space
        self.temperature = temperature
        self.supervised_simclr = supervised_simclr
        self.lr_opt = lr_opt

        self.model = S4Model(d_input=self.num_ifos,
                    length=self.num_timesteps,
                    d_output = self.d_output,
                    **s4_kwargs)

        self.projection_head = MLP(d_input = self.d_output, hidden_dims=[self.d_output], d_output = self.d_contrastive_space)
        
        self.loss_function = SupervisedSimCLRLoss(temperature=self.temperature,
                                                  contrast_mode='all', 
                                                  base_temperature=self.temperature)

        self.val_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr_opt)
        return optimizer

    def training_step(self, batch, batch_idx):
        if self.supervised_simclr:
            x,labels = batch
            x_embd = self.model(x)
            z_embd = self.projection_head(x_embd).unsqueeze(1) # shape (B,1,d_embd), just one "view" (no augmentations)
            z_embd = F.normalize(z_embd,dim=-1) # normalize for SimCLR loss
            loss = self.loss_function(z_embd,labels=labels)
        else:
            batch, labels = batch
            aug_0, aug_1 = batch[0], batch[1]
            z0 = self.model(aug_0)
            z1 = self.model(aug_1)
            embd_0 = self.projection_head(z0).unsqueeze(1)
            embd_1 = self.projection_head(z1).unsqueeze(1)
            embd_0 = F.normalize(embd_0,dim=-1) # normalize for SimCLR loss
            embd_1 = F.normalize(embd_1,dim=-1)
            embds = torch.cat((embd_0, embd_1), dim=1)
            loss = self.loss_function(embds,labels=None)

        self.metric = loss.detach()

        self.log(
            'train/loss',
            loss,
            sync_dist=True)

        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        if self.supervised_simclr:
            x,labels = batch
            x_embd = self.model(x)
            z_embd = self.projection_head(x_embd).unsqueeze(1) # shape (B,1,d_embd), just one "view" (no augmentations)
            z_embd = F.normalize(z_embd,dim=-1) # normalize for SimCLR loss
            loss = self.loss_function(z_embd,labels=labels)
        else:
            batch, labels = batch
            aug_0, aug_1 = batch[0], batch[1]
            z0 = self.model(aug_0)
            z1 = self.model(aug_1)
            embd_0 = self.projection_head(z0).unsqueeze(1)
            embd_1 = self.projection_head(z1).unsqueeze(1)
            embd_0 = F.normalize(embd_0,dim=-1) # normalize for SimCLR loss
            embd_1 = F.normalize(embd_1,dim=-1)
            embds = torch.cat((embd_0, embd_1), dim=1)
            loss = self.loss_function(embds,labels=None)

        self.log(
            'val/loss',
            loss,
            sync_dist=True)

        if self.supervised_simclr:
            self.val_outputs.append((loss, x_embd.cpu().numpy(), labels.cpu().numpy()))
        else:
            return loss
    
    def on_validation_epoch_end(self):
        if self.supervised_simclr:
            preds = np.concatenate([o[1] for o in self.val_outputs],axis=0)
            labels = np.concatenate([o[2] for o in self.val_outputs],axis=0)
            N = preds.shape[1]
            labs_uniq = sorted(list(set(labels)))
            fig,axes = plt.subplots(N,N,figsize=(20,20))

            for i in range(preds.shape[1]):
                for j in range(i+1,preds.shape[1]):
                    plt.sca(axes[i,j])
                    plt.axis('off')

            for i in range(preds.shape[1]):
                plt.sca(axes[i,i])
                plt.xticks([])
                plt.yticks([])
                bins = 30
                for j,lab in enumerate(labs_uniq):
                    h,bins,_ = plt.hist(preds[labels==lab][:,i],bins=bins,histtype='step',color=f"C{j}")
                    
            for i in range(1,preds.shape[1]):
                for j in range(i):
                    plt.sca(axes[i,j])
                    plt.xticks([])
                    plt.yticks([])
                    for k,lab in enumerate(labs_uniq):
                        ysel = preds[labels==lab]
                        plt.scatter(ysel[:,j],ysel[:,i],s=2,color=f"C{k}")
                        
            plt.sca(axes[0,2])
            patches = []
            for k,lab in enumerate(labs_uniq):
                patches.append(Patch(color=f"C{k}",label=f"Class {k+1}"))
            plt.legend(handles=patches,ncol=2,fontsize=12)

            buf = BytesIO()
            plt.savefig(buf,format='png')
            buf.seek(0)
            self.logger.log_image(
                'val/space',
                [Image.open(buf)],
            )

            plt.close(fig)

            self.val_outputs.clear()

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        # checkpoint for saving best model
        # that will be used for downstream export
        # and inference tasks
        # checkpoint for saving multiple best models
        callbacks = []

        # if using ray tune don't append lightning
        # model checkpoint since we'll be using ray's
        checkpoint = ModelCheckpoint(
            monitor='val/loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks

class EncoderTransformer(nn.Module):
    def __init__(self, num_timesteps:int=200,
                 num_features:int=2,
                 num_layers: int=4,
                 nhead: int=2,
                 latent_dim:int=16,
                 dim_factor_feedforward:int=4,
                 dropout:float=0.1,
                 embed_first:bool=True):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.dim_feedforward = dim_factor_feedforward*latent_dim
        self.dropout = dropout
        self.nhead = nhead
        self.embed_first = embed_first

        if self.embed_first:
            # linear embedding into the latent dimension
            self.embedding = nn.Linear(num_features, latent_dim, bias=False)

        # self-attention layers
        attn_layers = []
        for i in range(num_layers):
            attn_layers.append(nn.TransformerEncoderLayer(d_model=latent_dim,
                                                           nhead=nhead,
                                                           dim_feedforward=self.dim_feedforward,
                                                           dropout=dropout,
                                                           batch_first=True))
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x):
        x = x.transpose(1,2) # we get data in the shape (B,F,T) but transformers like (B,T,F)
        if self.embed_first:
            x = self.embedding(x)

        for layer in self.attn_layers:
            x = layer(x)

        return x
    
class ClassAttentionBlock(nn.Module):
    def __init__(self, num_timesteps:int=200,
                 dim:int=2,
                 nhead: int=2,
                 dropout:float=0.1,
                 dim_factor_feedforward:int=4,
                 scale_heads:bool=True,
                 scale_attn:bool=True,
                 scale_fc:bool=True,
                 scale_resids:bool=True,
                 ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.dim = dim
        self.dim_feedforward = dim_factor_feedforward * dim
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout) # shared dropout to use multiple places
        self.scale_heads = scale_heads
        self.scale_attn = scale_attn
        self.head_dim = dim // nhead

        # self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        # linear layers
        self.fc1 = nn.Linear(self.dim, self.dim_feedforward)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.dim_feedforward,self.dim)

        # layer norms
        self.pre_attn_norm = nn.LayerNorm(self.dim)
        self.post_attn_norm = nn.LayerNorm(self.dim) if scale_attn else None
        self.pre_fc_norm = nn.LayerNorm(self.dim)
        self.post_fc_norm = nn.LayerNorm(self.dim_feedforward) if scale_fc else None

        # attention head scaling and residual scaling
        self.c_attn = nn.Parameter(torch.ones(nhead), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(self.dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls):
        # x has shape (B,T,F) where F = num features, T = num timesteps
        # x_cls has shape (B,1,F) where F = num features
        
        # do self attention
        residual = x_cls
        u = torch.cat([x_cls,x],dim=1) # shape (B,T+1,F)
        u = self.pre_attn_norm(u)
        x = self.attn(x_cls,u,u,key_padding_mask=None,attn_mask=None,need_weights=False,is_causal=False)[0]

        # do attention head scaling if using
        if self.c_attn is not None:
            tgt_len = x.size(1) # sequence length i.e. 1, since it's just the class token
            x = x.view(-1, tgt_len, self.nhead, self.head_dim) # shape (B,1,H,F//H)
            x = torch.einsum('bthd,h->btdh', x, self.c_attn)
            x = x.reshape(-1, tgt_len, self.dim) # back to (B,1,F)

        # dropout and normalize
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        # feedforward layers
        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x
        
class ClassAttention(nn.Module):
    def __init__(self,dim,blocks:list[nn.Module]):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(blocks)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim), requires_grad=True) # define class token
        self.norm = nn.LayerNorm(self.dim)

    def forward(self,x):
        # x has shape (B,T,F)
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)
        for block in self.blocks:
            cls_tokens = block(x,cls_tokens)
        x = self.norm(cls_tokens).squeeze(1) # shape (B,F)
        return x

class Tarantula(GwakBaseModelClass):

    def __init__(
        self,
        num_ifos: int = 2,
        num_timesteps: int = 200,
        latent_dim: int = 64,
        num_layers: int = 4,
        num_head: int = 2,
        num_cls_layers: int = 2,
        fc_output_dims:list[int] = [],
        d_output:int = 16,
        d_contrastive_space: int = 16,
        normalize: bool = True,
        dropout: float = 0.1,
        cls_dropout: float = 0.0,
        feedforward_factor:int = 4,
        temperature: float = 0.1,
        supervised_simclr: bool = False,
        lr: float = 1e-4,
        min_lr: float = 1e-6,
        total_steps: int = None,
        warmup_fraction: float = 0.1,
        ):

        super().__init__()
        self.save_hyperparameters()

        self.num_ifos = num_ifos
        self.num_timesteps = num_timesteps
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_head = num_head
        self.num_cls_layers = num_cls_layers
        self.fc_output_dims = fc_output_dims
        self.d_output = d_output
        self.d_contrastive_space = d_contrastive_space
        self.normalize = normalize
        self.dropout = dropout
        self.cls_dropout = cls_dropout
        self.feedforward_factor = feedforward_factor
        self.temperature = temperature
        self.supervised_simclr = supervised_simclr
        self.lr = lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_fraction = warmup_fraction

        # define the self attention blocks
        self.self_attn = EncoderTransformer(
            num_timesteps=self.num_timesteps,
            num_features=self.num_ifos,
            num_layers=self.num_layers,
            nhead=self.num_head,
            latent_dim=self.latent_dim,
            dim_factor_feedforward=self.feedforward_factor,
            embed_first=True
        )

        # define the class attention blocks
        class_attn_blocks = []
        for i in range(self.num_cls_layers):
            class_attn_blocks.append(
                ClassAttentionBlock(
                    num_timesteps=self.num_timesteps,
                    dim=self.latent_dim,
                    nhead=self.num_head,
                    dropout=self.cls_dropout,
                    dim_factor_feedforward=self.feedforward_factor
                )
            )
        self.class_attn = ClassAttention(self.latent_dim,class_attn_blocks)

        # additional linear layer to project class token to d_output
        self.fc_out = MLP(d_input=self.latent_dim, hidden_dims=self.fc_output_dims, d_output=self.d_output)

        # define the model
        self.model = nn.Sequential(self.self_attn,self.class_attn,self.fc_out)

        # projection head with 1 hidden layer for SimCLR
        self.projection_head = MLP(d_input=self.d_output,
                                   hidden_dims=[self.d_output],
                                   d_output=self.d_contrastive_space
        )
        self.loss_function = SupervisedSimCLRLoss(temperature=self.temperature,
                                                  contrast_mode='all',
                                                  base_temperature=self.temperature)

    def training_step(self, batch, batch_idx):
        if self.supervised_simclr:
            x,labels = batch
            x_embd = self.model(x) # shape (B,1,d_embd)
            z_embd = self.projection_head(x_embd).unsqueeze(1) # shape (B,1,d_embd)
            z_embd = F.normalize(z_embd,dim=-1) # normalize for SimCLR loss
            self.metric = self.loss_function(z_embd,labels=labels)
        else:
            batch, labels = batch
            aug_0, aug_1 = batch[0], batch[1]
            z0 = self.model(aug_0)
            z1 = self.model(aug_1)
            embd_0 = self.projection_head(z0).unsqueeze(1)
            embd_1 = self.projection_head(z1).unsqueeze(1)
            embd_0 = F.normalize(embd_0,dim=-1)
            embd_1 = F.normalize(embd_1,dim=-1)
            embds = torch.cat((embd_0, embd_1), dim=1)
            self.metric = self.loss_function(embds,labels=None)

        self.log(
            'train/loss',
            self.metric,
            sync_dist=True)

        return self.metric

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        if self.supervised_simclr:
            x,labels = batch
            x_embd = self.model(x)
            z_embd = self.projection_head(x_embd).unsqueeze(1) # shape (B,1,d_embd), just one "view" (no augmentations)
            z_embd = F.normalize(z_embd,dim=-1) # normalize for SimCLR loss
            loss = self.loss_function(z_embd,labels=labels)
        else:
            batch, labels = batch
            aug_0, aug_1 = batch[0], batch[1]
            z0 = self.model(aug_0)
            z1 = self.model(aug_1)
            embd_0 = self.projection_head(z0).unsqueeze(1)
            embd_1 = self.projection_head(z1).unsqueeze(1)
            embd_0 = F.normalize(embd_0,dim=-1) # normalize for SimCLR loss
            embd_1 = F.normalize(embd_1,dim=-1)
            embds = torch.cat((embd_0, embd_1), dim=1)
            loss = self.loss_function(embds,labels=None)

        self.log(
            'val/loss',
            loss,
            sync_dist=True)

        return loss
    
    @torch.no_grad
    def test_step(self,batch,batch_idx):
        x,labels = batch
        x_embd = self.model(x).detach().cpu().numpy()
        x_proj = self.projection_head(x_embd).detach().cpu().numpy()

        output = {
            "embeddings": x_embd,
            "projections": x_proj,
            "labels": labels.detach().cpu().numpy()
        }

        return output
    
    def test_epoch_end(self,outputs):
        output_arrays = {k:[] for k in outputs[0].keys()}
        for o in outputs:
            for k,v in o.items():
                output_arrays[k].append(v)
        for k in output_arrays.keys():
            output_arrays[k] = np.concatenate(output_arrays[k],axis=0)
        
        outdir = self.trainer.logger.save_dir
        with h5py.File(f"{outdir}/test_embeddings.h5","w") as fout:
            for k,v in output_arrays.items():
                fout.create_dataset(k,data=v)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        scheduler = WarmupCosineAnnealingLR(optimizer,self.total_steps,self.lr,self.min_lr,self.warmup_fraction)
        return {
            "optimizer":optimizer,
            "lr_scheduler": {
                "scheduler":scheduler,
                "interval":"step",
                "frequency":1
            }
        }
        


    def configure_callbacks(self) -> Sequence[pl.Callback]:
        # checkpoint for saving best model
        # that will be used for downstream export
        # and inference tasks
        # checkpoint for saving multiple best models
        callbacks = []

        # if using ray tune don't append lightning
        # model checkpoint since we'll be using ray's
        checkpoint = ModelCheckpoint(
            monitor='val/loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks
    