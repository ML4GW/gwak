import os
import time
import logging
from typing import Sequence
from collections import OrderedDict

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
            arch=pl_module.model, 
            metric=pl_module.metric
        )
        
        # Modifiy these to read the last training/valdation data 
        # to acess the input shape. 
        # X = torch.randn(1, 200, 2) # GWAK 1
        X = torch.randn(1, 2, 200) # GWAK 2

        trace = torch.jit.trace(module.model.to("cpu"), X.to("cpu"))

        save_dir = trainer.logger.log_dir or trainer.logger.save_dir

        with open(os.path.join(save_dir, "model_JIT.pt"), "wb") as f:
            torch.jit.save(trace, f)


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
        lr: Optional[float] = None,
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

class ProjectionHeadModel(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        #copying the paper of having one-layer MLP
        self.d_input = d_input
        self.d_output = d_output

        self.layer = nn.Linear(d_input, d_output)

    def forward(self, x):
        return F.relu(self.layer(x))


class Crayon(GwakBaseModelClass):

    def __init__(
        self,
        num_ifos: int = 2,
        num_timesteps: int = 200,
        d_output:int = 10,
        d_contrastive_space: int = 20,
        temperature: float = 0.1
        ):

        super().__init__()
        self.num_ifos = num_ifos
        self.num_timesteps = num_timesteps
        self.d_output = d_output
        self.d_contrastive_space = d_contrastive_space

        self.temperature = temperature

        self.model = S4Model(d_input=self.num_ifos,
                    length=self.num_timesteps,
                    d_output = self.d_output)
        
        self.projection_head = ProjectionHeadModel(d_input = self.d_output,
                                                    d_output = self.d_contrastive_space)

    def simCLR(self, z0, z1):
        N = len(z0)
        z = torch.stack((z0, z1), dim=1).reshape(-1, z0.shape[1]) #intertwine
        z_norm = z / torch.norm(z, dim=1)[:, None]
        Sij = z_norm @ torch.transpose(z_norm, 0, 1) / self.temperature
        Sij_exp = torch.exp( Sij )
        denom_k = torch.sum( Sij_exp, dim=1 ) - torch.diagonal(Sij_exp)
        numerator_k = torch.diag(Sij, diagonal=-1)[::2]
        return 1/(2*N) * ( -2 * torch.sum(numerator_k) + torch.sum(torch.log(denom_k))  )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())#, lr=config.learning_rate
        return optimizer
        
    def training_step(self, batch, batch_idx):
        aug_0, aug_1 = batch[0], batch[1]
        embd_0 = self.projection_head(self.model(aug_0))
        embd_1 = self.projection_head(self.model(aug_1))

        self.metric = self.simCLR(embd_0, embd_1)
        #loss = self.simCLR(embd_0, embd_1)

        self.log(
            'train_loss',
            self.metric,
            sync_dist=True)

        return self.metric

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        aug_0, aug_1 = batch[0], batch[1]
        embd_0 = self.projection_head(self.model(aug_0))
        embd_1 = self.projection_head(self.model(aug_1))

        loss = self.simCLR(embd_0, embd_1)

        self.log(
            'val_loss',
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
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks
    
class EncoderTransformer(nn.Module):
    def __init__(self, num_timesteps:int=200, num_features:int=2, latent_dim:int=16):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.latent_dim = latent_dim

        dim_feedforward = 128
        nhead=2
        self.transformer1 = nn.TransformerEncoderLayer(d_model=num_features,  
                                                       nhead=nhead, 
                                                       dim_feedforward=dim_feedforward,
                                                       batch_first=True)
        self.layer1 = nn.Linear(num_features, num_features*2)
        self.transformer2 = nn.TransformerEncoderLayer(d_model=num_features*2,  nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.layer2 = nn.Linear(num_features*2, num_features*2)
        self.transformer3 = nn.TransformerEncoderLayer(d_model=num_features*2,  nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.layer3 = nn.Linear(num_timesteps*(num_features*2), latent_dim * 4)
        self.layer4 = nn.Linear(latent_dim*4, latent_dim*4)
        self.layer5 = nn.Linear(latent_dim*4, latent_dim)

    def forward(self, x):
        num_batches, num_ifos, num_timesteps = x.shape

        x = x.reshape(num_batches, num_timesteps, num_ifos)
        x = self.transformer1(x)
        x = F.relu(self.layer1(x))

        x = self.transformer2(x)
        x = F.relu(self.layer2(x))
        
        x = self.transformer3(x)
        #print(772000000, x.shape)
        x = x.reshape( (num_batches, self.num_timesteps * (self.num_features*2)))

        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)

        return x
    
class Tarantula(GwakBaseModelClass):

    def __init__(
        self,
        num_ifos: int = 2,
        num_timesteps: int = 200,
        d_output:int = 16,
        d_contrastive_space: int = 16,
        temperature: float = 0.1
        ):

        super().__init__()

        self.num_ifos = num_ifos
        self.num_timesteps = num_timesteps
        self.d_output = d_output
        self.d_contrastive_space = d_contrastive_space

        self.temperature = temperature
        self.model = EncoderTransformer(num_timesteps = self.num_timesteps,
                                          num_features = self.num_ifos,
                                          latent_dim = self.d_output)
        
        
        self.projection_head = ProjectionHeadModel(d_input = self.d_output,
                                                    d_output = self.d_contrastive_space)

    def simCLR(self, z0, z1):
        N = len(z0)
        z = torch.stack((z0, z1), dim=1).reshape(-1, z0.shape[1]) #intertwine
        z_norm = z / torch.norm(z, dim=1)[:, None]
        Sij = z_norm @ torch.transpose(z_norm, 0, 1) / self.temperature
        Sij_exp = torch.exp( Sij )
        denom_k = torch.sum( Sij_exp, dim=1 ) - torch.diagonal(Sij_exp)
        numerator_k = torch.diag(Sij, diagonal=-1)[::2]
        return 1/(2*N) * ( -2 * torch.sum(numerator_k) + torch.sum(torch.log(denom_k))  )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())#, lr=config.learning_rate
        return optimizer
        
    def training_step(self, batch, batch_idx):
        aug_0, aug_1 = batch[0], batch[1]
        z0 = self.model(aug_0)
        z1 = self.model(aug_1)
        embd_0 = self.projection_head(z0)
        embd_1 = self.projection_head(z1)

        self.metric = self.simCLR(embd_0, embd_1)


        self.log(
            'train_loss',
            self.metric,
            sync_dist=True)

        return self.metric

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        aug_0, aug_1 = batch[0], batch[1]
        z0 = self.model(aug_0)
        z1 = self.model(aug_1)
        embd_0 = self.projection_head(z0)
        embd_1 = self.projection_head(z1)
        loss = self.simCLR(embd_0, embd_1)

        self.log(
            'val_loss',
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
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks
    