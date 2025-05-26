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

class EncoderTransformer(nn.Module):
    def __init__(self, num_timesteps:int=200,
                 num_features:int=2,
                 num_layers: int=4,
                 nhead: int=2,
                 latent_dim:int=16,
                 dim_factor_feedforward:int=4,
                 dropout:float=0.1,
                 embed_first:bool=True,
                 patch_size:int=None):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.dim_feedforward = dim_factor_feedforward*latent_dim
        self.dropout = dropout
        self.nhead = nhead
        self.embed_first = embed_first
        self.patch_size = patch_size

        # Check if patching is enabled and validate patch size
        if self.patch_size is not None:
            assert num_timesteps % patch_size == 0, f"Patch size {patch_size} must divide the number of timesteps {num_timesteps}"
            self.num_patches = num_timesteps // patch_size
            # Patch embedding layer (maps each patch to latent_dim)
            self.patch_embedding = nn.Linear(num_features * patch_size, latent_dim)
        elif self.embed_first:
            # linear embedding into the latent dimension (no patching)
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
        # Input shape: (B, F, T) - batch, features, timesteps
        
        if self.patch_size is not None:
            batch_size = x.size(0)
            
            # Reshape for patching: (B, F, T) -> (B, F, num_patches, patch_size)
            x = x.reshape(batch_size, self.num_features, self.num_patches, self.patch_size)
            
            # Transpose to get (B, num_patches, F, patch_size)
            x = x.permute(0, 2, 1, 3)
            
            # Flatten patches: (B, num_patches, F*patch_size)
            x = x.reshape(batch_size, self.num_patches, self.num_features * self.patch_size)
            
            # Embed each patch
            x = self.patch_embedding(x)  # (B, num_patches, latent_dim)
        else:
            x = x.transpose(1, 2)  # (B, F, T) -> (B, T, F)
            if self.embed_first:
                x = self.embedding(x)  # (B, T, latent_dim)

        # Apply transformer layers
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
                 patch_size:int=None):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.dim = dim
        self.dim_feedforward = dim_factor_feedforward * dim
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout) # shared dropout to use multiple places
        self.scale_heads = scale_heads
        self.scale_attn = scale_attn
        self.head_dim = dim // nhead
        self.patch_size = patch_size

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
        # x has shape (B,T',F) where F = num features, T' = num timesteps or num patches
        # x_cls has shape (B,1,F) where F = num features
        
        # do self attention
        residual = x_cls
        u = torch.cat([x_cls,x],dim=1) # shape (B,T'+1,F)
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