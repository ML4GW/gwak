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
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from gwak.train.losses import SupervisedSimCLRLoss
from gwak.train.schedulers import WarmupCosineAnnealingLR
from gwak.train.plotting import make_corner

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import wandb
from PIL import Image
from io import BytesIO
import shutil

from transformers import EncoderTransformer, ClassAttentionBlock, ClassAttention, InvertedEncoderTransformer
from ssm import DropoutNd, S4DKernel, S4D, S4Model
from nets import MLP, Encoder, Decoder
from callback import ModelCheckpoint
from resnet_1d import ResNet1D


class GwakBaseModelClass(pl.LightningModule):

    def get_logger(self):
        logger_name = 'GwakBaseModelClass'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        return logger

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

class SimCLRBase(GwakBaseModelClass):
    def get_temperature(self,epoch):
        if self.temperature_init is None:
            return self.temperature
        else:
            if epoch < self.n_temp_anneal:
                m = (self.temperature - self.temperature_init) / self.n_temp_anneal
                return self.temperature_init + m * epoch
            else:
                return self.temperature
        
    def get_lambda_classifier(self,epoch):
        if self.anneal_classifier:
            if epoch < self.class_anneal_epochs:
                return self.lambda_classifier_original * ((self.class_anneal_epochs - epoch) / self.class_anneal_epochs)
            else:
                return 0.0
        else:
            return self.lambda_classifier_original
    
    def on_train_epoch_start(self):
        #self.get_logger().info(f"Train epoch start called, epoch {self.current_epoch}")
        temp = self.get_temperature(self.current_epoch)
        lambda_class = self.get_lambda_classifier(self.current_epoch)
        self.log("train/temperature", temp)
        if self.use_classifier:
            self.log("train/lambda_class", lambda_class)
        self.loss_function = SupervisedSimCLRLoss(temperature=temp,
                                                  contrast_mode='all', 
                                                  base_temperature=temp)
        self.lambda_classifier = lambda_class

    def training_step(self, batch, batch_idx):
        if self.supervised_simclr:
            x,labels = batch
            x_embd = self.model(x)
            z_embd = self.projection_head(x_embd).unsqueeze(1) # shape (B,1,d_embd), just one "view" (no augmentations)
            z_embd = F.normalize(z_embd,dim=-1) # normalize for SimCLR loss
            loss_simclr = self.loss_function(z_embd,labels=labels)
            if self.use_classifier:
                logits = self.classifier(x_embd)
                #self.get_logger().info(f"Logits dtype: {logits.dtype}")
                #self.get_logger().info(f"labels dtype: {labels.dtype}")
                loss_class = F.cross_entropy(logits, (labels-1).to(torch.long))
                loss = loss_simclr + self.lambda_classifier * loss_class
            else:
                loss = loss_simclr
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
            loss_simclr = self.loss_function(embds,labels=None)
            loss = loss_simclr

        self.metric = loss.detach()

        self.log(
            'train/loss',
            loss,
            sync_dist=True)
        self.log(
            'train/loss_simclr',
            loss_simclr,
            sync_dist=True)
        if self.use_classifier:
            self.log(
                'train/loss_classifier',
                loss_class,
                sync_dist=True)

        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        if self.supervised_simclr:
            x,labels = batch
            x_embd = self.model(x)
            z_embd = self.projection_head(x_embd).unsqueeze(1) # shape (B,1,d_embd), just one "view" (no augmentations)
            z_embd = F.normalize(z_embd,dim=-1) # normalize for SimCLR loss
            loss_simclr = self.loss_function(z_embd,labels=labels)
            if self.use_classifier:
                logits = self.classifier(x_embd)
                #self.get_logger().info(f"Logits dtype: {logits.dtype}")
                #self.get_logger().info(f"labels dtype: {labels.dtype}")
                #self.get_logger().info(f"Logits shape: {logits.shape}")
                #self.get_logger().info(f"labels shape: {labels.shape}")
                loss_class = F.cross_entropy(logits, (labels-1).to(torch.long))
                loss = loss_simclr + self.lambda_classifier * loss_class
            else:
                loss = loss_simclr
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
            loss_simclr = self.loss_function(embds,labels=None)
            loss = loss_simclr

        self.log(
            'val/loss',
            loss,
            sync_dist=True)
        self.log(
            'val/loss_simclr',
            loss_simclr,
            sync_dist=True)
        if self.use_classifier:
            self.log(
                'val/loss_classifier',
                loss_class,
                sync_dist=True)

        if self.supervised_simclr:
            self.val_outputs.append((loss.item(), x_embd.cpu().numpy(), labels.cpu().numpy()))
        else:
            return loss

    def on_validation_epoch_end(self):
        if self.supervised_simclr:
            preds = np.concatenate([o[1] for o in self.val_outputs],axis=0)
            labels = np.concatenate([o[2] for o in self.val_outputs],axis=0)
            #sig_classes = self.trainer.datamodule.signal_classes
            #label_names = {i+1:c for i,c in enumerate(sig_classes)}
            label_names = self.trainer.datamodule.all_signal_label_names
            fig = make_corner(preds,labels,return_fig=True,label_names=label_names)

            buf = BytesIO()
            fig.savefig(buf,format='jpg',dpi=200)
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

class Crayon(SimCLRBase):

    def __init__(
        self,
        num_ifos: Union[int,str] = 2,
        num_timesteps: int = 200,
        d_output:int = 10,
        d_contrastive_space: int = 20,
        temperature: float = 0.1,
        temperature_init: float = None, # initial temperature to start with, if None then constant temp throughout
        n_temp_anneal: int = 10, # number of epochs to anneal temperature
        supervised_simclr: bool = False,
        lr_opt: float = 1e-4,
        lr_min: float = 1e-5,
        cos_anneal: bool = False,
        cos_anneal_tmax: int = 50,
        num_classes: int = 8,
        use_classifier: bool = False,
        lambda_classifier: float = 0.5,
        class_anneal_epochs: int = 10, # number of epochs over which to anneal lambda_classifier
        anneal_classifier: bool = True, # whether to anneal classifier loss
        s4_kwargs: Optional[dict] = {},
        classifier_hidden_dims: Optional[list[int]] = None,
        ):

        super().__init__()
        self.num_ifos = num_ifos if type(num_ifos) == int else len(num_ifos)
        self.num_timesteps = num_timesteps
        self.d_output = d_output
        self.d_contrastive_space = d_contrastive_space
        self.temperature = temperature
        self.temperature_init = temperature_init
        self.n_temp_anneal = n_temp_anneal
        self.supervised_simclr = supervised_simclr
        self.lr_opt = lr_opt
        self.lr_min = lr_min
        self.cos_anneal = cos_anneal
        self.cos_anneal_tmax = cos_anneal_tmax
        self.num_classes = num_classes
        self.use_classifier = use_classifier
        self.lambda_classifier = lambda_classifier
        self.lambda_classifier_original = lambda_classifier
        self.class_anneal_epochs = class_anneal_epochs
        self.anneal_classifier = anneal_classifier
        self.classifier_hidden_dims = classifier_hidden_dims

        self.model = S4Model(d_input=self.num_ifos,
                    length=self.num_timesteps,
                    d_output = self.d_output,
                    **s4_kwargs)

        self.projection_head = MLP(d_input = self.d_output, hidden_dims=[self.d_output], d_output = self.d_contrastive_space)
        if self.use_classifier:
            hidden_dims = self.classifier_hidden_dims if self.classifier_hidden_dims is not None else [4*self.d_output for _ in range(2)]
            self.classifier = MLP(d_input = self.d_output, hidden_dims=hidden_dims, 
                                d_output = self.num_classes)
        
        self.loss_function = SupervisedSimCLRLoss(temperature=self.temperature_init if self.temperature_init is not None else self.temperature,
                                                  contrast_mode='all', 
                                                  base_temperature=self.temperature)

        self.val_outputs = []

        self.save_hyperparameters()

    def configure_optimizers(self):
        all_parameters = list(self.parameters())
        general_params = [p for p in all_parameters if not hasattr(p, "_optim")]
        optimizer = torch.optim.AdamW(general_params,lr=self.lr_opt)

        # special parameters for S4 that get their own LR
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        #self.get_logger().info("doing unique stuff with optimizers")
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )
        
        if self.cos_anneal:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cos_anneal_tmax, eta_min=self.lr_min)
            return {"optimizer": optimizer, "lr_scheduler": sched}
        else:
            return optimizer

class Tarantula(SimCLRBase):

    def __init__(
        self,
        num_ifos: Union[int,str] = 2,
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
        patch_size: int = None,
        use_classifier: bool = False,
        num_classes: int = 8,
        lambda_classifier: float = 0.5,
        class_anneal_epochs: int = 10, # number of epochs over which to anneal lambda_classifier
        anneal_classifier: bool = True, # whether to anneal classifier loss
        temperature_init: Optional[float] = None, # initial temperature to start with, if None then constant temp throughout
        n_temp_anneal: int = 10, # number of epochs to anneal temperature
        classifier_hidden_dims: Optional[list[int]] = None
        ):

        super().__init__()
        self.save_hyperparameters()

        self.num_ifos = num_ifos if type(num_ifos) == int else len(num_ifos)
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
        self.patch_size = patch_size
        self.use_classifier = use_classifier
        self.num_classes = num_classes
        self.lambda_classifier = lambda_classifier
        self.lambda_classifier_original = lambda_classifier
        self.class_anneal_epochs = class_anneal_epochs
        self.anneal_classifier = anneal_classifier
        self.temperature = temperature
        self.temperature_init = temperature_init
        self.n_temp_anneal = n_temp_anneal
        self.classifier_hidden_dims = classifier_hidden_dims
        
        # define the self attention blocks
        self.self_attn = EncoderTransformer(
            num_features=self.num_ifos,
            num_layers=self.num_layers,
            nhead=self.num_head,
            latent_dim=self.latent_dim,
            dim_factor_feedforward=self.feedforward_factor,
            embed_first=True,
            patch_size=self.patch_size
        )

        # define the class attention blocks
        class_attn_blocks = []
        for i in range(self.num_cls_layers):
            class_attn_blocks.append(
                ClassAttentionBlock(
                    dim=self.latent_dim,
                    nhead=self.num_head,
                    dropout=self.cls_dropout,
                    dim_factor_feedforward=self.feedforward_factor
                )
            )
        self.class_attn = ClassAttention(self.latent_dim, class_attn_blocks)

        # additional linear layer to project class token to d_output
        self.fc_out = MLP(d_input=self.latent_dim, hidden_dims=self.fc_output_dims, d_output=self.d_output)

        # define the model
        self.model = nn.Sequential(self.self_attn,self.class_attn,self.fc_out)

        # projection head with 1 hidden layer for SimCLR
        self.projection_head = MLP(d_input=self.d_output,
                                   hidden_dims=[self.d_output],
                                   d_output=self.d_contrastive_space
        )
        if self.use_classifier:
            hidden_dims = [4*self.d_output for _ in range(2)] if self.classifier_hidden_dims is None else self.classifier_hidden_dims
            self.classifier = MLP(d_input = self.d_output, hidden_dims=hidden_dims, 
                                d_output = self.num_classes)
            
        self.loss_function = SupervisedSimCLRLoss(temperature=self.temperature_init if self.temperature_init is not None else self.temperature,
                                                  contrast_mode='all', 
                                                  base_temperature=self.temperature)

        self.val_outputs = []
        self.save_hyperparameters()

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
    
class Contour(SimCLRBase):

    def __init__(
        self,
        num_ifos: Union[int,str] = 2,
        d_output:int = 10,
        d_contrastive_space: int = 20,
        resnet_layers: list[int] = [3, 4, 6, 3], # ResNet-18
        resnet_kernel_size: int = 3, # kernel size for ResNet-1D
        temperature: float = 0.1,
        temperature_init: float = None, # initial temperature to start with, if None then constant temp throughout
        n_temp_anneal: int = 10, # number of epochs to anneal temperature
        supervised_simclr: bool = False,
        lr: float = 1e-4,
        lr_min: float = 1e-5,
        cos_anneal: bool = False,
        cos_anneal_tmax: int = 50,
        num_classes: int = 8,
        use_classifier: bool = False,
        lambda_classifier: float = 0.5,
        class_anneal_epochs: int = 10, # number of epochs over which to anneal lambda_classifier
        anneal_classifier: bool = True, # whether to anneal classifier loss
        classifier_hidden_dims: Optional[list[int]] = None,
        projector_hidden_dims: Optional[list[int]] = [64,64],
        ):

        super().__init__()
        self.num_ifos = num_ifos if type(num_ifos) == int else len(num_ifos)
        self.d_output = d_output
        self.d_contrastive_space = d_contrastive_space
        self.resnet_layers = resnet_layers
        self.resnet_kernel_size = resnet_kernel_size
        self.temperature = temperature
        self.temperature_init = temperature_init
        self.n_temp_anneal = n_temp_anneal
        self.supervised_simclr = supervised_simclr
        self.lr = lr
        self.lr_min = lr_min
        self.cos_anneal = cos_anneal
        self.cos_anneal_tmax = cos_anneal_tmax
        self.num_classes = num_classes
        self.use_classifier = use_classifier
        self.lambda_classifier = lambda_classifier
        self.lambda_classifier_original = lambda_classifier
        self.class_anneal_epochs = class_anneal_epochs
        self.anneal_classifier = anneal_classifier
        self.classifier_hidden_dims = classifier_hidden_dims

        self.model = ResNet1D(
            in_channels=self.num_ifos,
            classes=self.d_output,
            layers=self.resnet_layers,
            kernel_size=self.resnet_kernel_size
        )

        self.projector_hidden_dims = projector_hidden_dims if projector_hidden_dims is not None else [4*self.d_output, 4*self.d_output]
        self.projection_head = MLP(d_input = self.d_output, hidden_dims=self.projector_hidden_dims, d_output = self.d_contrastive_space)
        if self.use_classifier:
            hidden_dims = self.classifier_hidden_dims if self.classifier_hidden_dims is not None else [4*self.d_output for _ in range(2)]
            self.classifier = MLP(d_input = self.d_output, hidden_dims=hidden_dims, 
                                d_output = self.num_classes)
        
        self.loss_function = SupervisedSimCLRLoss(temperature=self.temperature_init if self.temperature_init is not None else self.temperature,
                                                  contrast_mode='all', 
                                                  base_temperature=self.temperature)

        self.val_outputs = []

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        if self.cos_anneal:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cos_anneal_tmax, eta_min=self.lr_min)
            return {"optimizer": optimizer, "lr_scheduler": sched}
        else:
            return optimizer
        
class iTransformer(SimCLRBase):
    def __init__(
        self,
        num_ifos: Union[int,str] = 2,
        num_timesteps: int = 4096,
        latent_dim: int = 128,
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
        use_classifier: bool = False,
        num_classes: int = 8,
        lambda_classifier: float = 0.5,
        class_anneal_epochs: int = 10, # number of epochs over which to anneal lambda_classifier
        anneal_classifier: bool = True, # whether to anneal classifier loss
        temperature_init: Optional[float] = None, # initial temperature to start with, if None then constant temp throughout
        n_temp_anneal: int = 10, # number of epochs to anneal temperature
        classifier_hidden_dims: Optional[list[int]] = None
        ):

        super().__init__()
        self.save_hyperparameters()

        self.num_ifos = num_ifos if type(num_ifos) == int else len(num_ifos)
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
        self.use_classifier = use_classifier
        self.num_classes = num_classes
        self.lambda_classifier = lambda_classifier
        self.lambda_classifier_original = lambda_classifier
        self.class_anneal_epochs = class_anneal_epochs
        self.anneal_classifier = anneal_classifier
        self.temperature = temperature
        self.temperature_init = temperature_init
        self.n_temp_anneal = n_temp_anneal
        self.classifier_hidden_dims = classifier_hidden_dims
        
        # define the self attention blocks
        self.self_attn = InvertedEncoderTransformer(
            num_features=self.num_timesteps,
            num_layers=self.num_layers,
            nhead=self.num_head,
            latent_dim=self.latent_dim,
            dim_factor_feedforward=self.feedforward_factor,
            embed_first=True,
        )

        # define the class attention blocks
        class_attn_blocks = []
        for i in range(self.num_cls_layers):
            class_attn_blocks.append(
                ClassAttentionBlock(
                    dim=self.latent_dim,
                    nhead=self.num_head,
                    dropout=self.cls_dropout,
                    dim_factor_feedforward=self.feedforward_factor
                )
            )
        self.class_attn = ClassAttention(self.latent_dim, class_attn_blocks)

        # additional linear layer to project class token to d_output
        self.fc_out = MLP(d_input=self.latent_dim, hidden_dims=self.fc_output_dims, d_output=self.d_output)

        # define the model
        self.model = nn.Sequential(self.self_attn,self.class_attn,self.fc_out)

        # projection head with 1 hidden layer for SimCLR
        self.projection_head = MLP(d_input=self.d_output,
                                   hidden_dims=[self.d_output],
                                   d_output=self.d_contrastive_space
        )
        if self.use_classifier:
            hidden_dims = [4*self.d_output for _ in range(2)] if self.classifier_hidden_dims is None else self.classifier_hidden_dims
            self.classifier = MLP(d_input = self.d_output, hidden_dims=hidden_dims, 
                                d_output = self.num_classes)
            
        self.loss_function = SupervisedSimCLRLoss(temperature=self.temperature_init if self.temperature_init is not None else self.temperature,
                                                  contrast_mode='all', 
                                                  base_temperature=self.temperature)

        self.val_outputs = []
        self.save_hyperparameters()

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