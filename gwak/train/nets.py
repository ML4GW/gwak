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

class MLP(nn.Module):
    def __init__(self, d_input:int, hidden_dims:list[int], d_output, dropout=0.0, activation=nn.SiLU(), output_activation=None):
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