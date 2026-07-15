import os
import time
import yaml
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning.pytorch as pl

import math
from typing import Optional, Union, Sequence

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim:int, bias:bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input:Tensor):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# manual attn implementation for when we need to use it (e.g. non-vanilla transformers)
class SelfAttention(nn.Module):

    def __init__(self, d_embd:int, n_head:int, dropout:float=0.1, bias:bool=False):
        super().__init__()
        assert d_embd % n_head == 0
        self.d_embd = d_embd
        self.n_head = n_head
        self.dropout = dropout
        self.n_embd_head = d_embd // n_head
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_embd, 3 * d_embd, bias=bias)
        
        # output projection
        self.c_proj = nn.Linear(d_embd, d_embd, bias=bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # attribute to store attention mask, so we can retrieve it if we want
        self.attn_map = None

    def forward(self, x:Tensor):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d_embd, dim=2)
        k = k.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class FanMLP(nn.Module):

    def __init__(self, d_embd:int, fan_factor:int=4, dropout:float=0.1, bias:bool=False):
        super().__init__()
        self.c_fc    = nn.Linear(d_embd, fan_factor * d_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(fan_factor * d_embd, d_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class FanConv(nn.Module):
    def __init__(self, d_embd:int, fan_factor:int=4, dropout:float=0.1, bias:bool=False):
        super().__init__()
        self.c_conv = nn.Conv1d(d_embd, fan_factor * d_embd, kernel_size=1, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Conv1d(fan_factor * d_embd, d_embd, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor):
        x = self.c_conv(x.transpose(1, 2))  # Conv1d expects time-dimension last
        x = self.gelu(x)
        x = self.c_proj(x).transpose(1, 2)  # Back to normal transformer ordering
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, d_embd:int, n_head:int, fan_factor:int=4, dropout:float=0.1, bias:bool=False, use_conv:bool=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_embd, bias=bias)
        self.attn = SelfAttention(d_embd, n_head, dropout=dropout, bias=bias)
        self.ln_2 = LayerNorm(d_embd, bias=bias)
        self.mlp = FanConv(d_embd, fan_factor=fan_factor, dropout=dropout, bias=bias) if use_conv else FanMLP(d_embd, fan_factor=fan_factor, dropout=dropout, bias=bias)

    def forward(self, x:Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class EncoderTransformer(nn.Module):
    def __init__(self,
                 num_features:int=2,
                 num_layers: int=4,
                 nhead: int=2,
                 latent_dim:int=16,
                 dim_factor_feedforward:int=4,
                 dropout:float=0.1,
                 embed_first:bool=True,
                 patch_size:int=None):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.dim_feedforward = dim_factor_feedforward*latent_dim
        self.dropout = dropout
        self.nhead = nhead
        self.embed_first = embed_first
        self.patch_size = patch_size

        # Check if patching is enabled and validate patch size
        if self.patch_size is not None:
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

    def forward(self, x:Tensor):
        # Input shape: (B, F, T) - batch, features, timesteps (assuming LIGO-style)
        
        if self.patch_size is not None:
            batch_size = x.size(0)
            num_timesteps = x.size(2)
            num_patches = num_timesteps // self.patch_size
            
            # Reshape for patching: (B, F, T) -> (B, F, num_patches, patch_size)
            x = x.reshape(batch_size, self.num_features, num_patches, self.patch_size)
            
            # Transpose to get (B, num_patches, F, patch_size)
            x = x.permute(0, 2, 1, 3)
            
            # Flatten patches: (B, num_patches, F*patch_size)
            x = x.reshape(batch_size, num_patches, self.num_features * self.patch_size)
            
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
    
class InvertedEncoderTransformer(nn.Module):
    # inspired by iTransformer https://arxiv.org/abs/2310.06625
    # idea is that you embed the *entire time series* of each channel (i.e. ifo) into d_model, then do self-attention between the embeddings of the channels
    # e.g. there will only be 2-3 "tokens" in the sequence, one for each channel (ifo)
    # maybe this will facilitate easier learning of correlations between ifos?
    def __init__(self,
                 num_features:int=4096,
                 num_ifos:int=2,
                 num_layers: int=4,
                 nhead: int=2,
                 latent_dim:int=512,
                 conv_embd_kernel_size:int=64,
                 conv_embd_channels:int=16,
                 dim_factor_feedforward:int=4,
                 dropout:float=0.1,
                 bias: bool=False,
                 use_conv: bool=True,
                 time_last: bool=True,
                 embed_first:bool=True):
        super().__init__()
        self.num_features = num_features
        self.num_ifos = num_ifos
        self.latent_dim = latent_dim
        self.dim_factor_feedforward = dim_factor_feedforward
        self.dropout = dropout
        self.nhead = nhead
        self.bias = bias
        self.use_conv = use_conv
        self.embed_first = embed_first
        self.time_last = time_last

        if self.embed_first:
            # linear embedding into the latent dimension (no patching)
            #self.embedding = nn.Linear(num_features, latent_dim, bias=False)
            # use conv embedding instead?
            modules = [nn.Conv1d(in_channels = num_ifos, out_channels = conv_embd_channels, 
                                 kernel_size = conv_embd_kernel_size, stride = 1, padding = 'same', bias = True),
                        nn.BatchNorm1d(conv_embd_channels),
                        nn.SiLU(),
                        nn.AdaptiveAvgPool1d(latent_dim)]  # Forces exact output length
            self.embedding = nn.Sequential(*modules)

        # self-attention layers
        attn_layers = []
        for i in range(num_layers):
            attn_layers.append(Block(d_embd=latent_dim,
                                     n_head=nhead,
                                     fan_factor=self.dim_factor_feedforward,
                                     dropout=dropout,
                                     bias=bias,
                                     use_conv=use_conv))
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x:Tensor):
        # Input shape: (B, F, T) - batch, features, timesteps (LIGO-style)
        # we are actually embedding the time dimension, so it doesn't need to be transposed
        if not self.time_last:
            x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)

        if self.embed_first:
            x = self.embedding(x)  # (B, F, latent_dim)

        # Apply transformer layers
        for layer in self.attn_layers:
            x = layer(x)

        return x

class TransformerEmbedder(nn.Module):
    def __init__(self, transformer:nn.Module, mlp:nn.Module):
        super().__init__()
        self.transformer = transformer
        self.mlp = mlp
    
    def forward(self, x:Tensor):
        x = self.transformer(x)  # (B, T, latent_dim)
        x = x.mean(dim=1)  # Global average pooling over time dimension (B, latent_dim)
        x = self.mlp(x)  # (B, output_dim)
        return x
    
class ClassAttentionBlock(nn.Module):
    def __init__(self,
                 dim:int=2,
                 nhead: int=2,
                 dropout:float=0.1,
                 dim_factor_feedforward:int=4,
                 scale_heads:bool=True,
                 scale_attn:bool=True,
                 scale_fc:bool=True,
                 scale_resids:bool=True):
        super().__init__()
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

    def forward(self, x:Tensor, x_cls:Tensor):
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
    def __init__(self,dim:int,blocks:list[nn.Module]):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(blocks)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim), requires_grad=True) # define class token
        self.norm = nn.LayerNorm(self.dim)

    def forward(self,x:Tensor):
        # x has shape (B,T,F)
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)
        for block in self.blocks:
            cls_tokens = block(x,cls_tokens)
        x = self.norm(cls_tokens).squeeze(1) # shape (B,F)
        return x