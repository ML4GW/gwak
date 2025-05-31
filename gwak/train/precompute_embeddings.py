import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
import lightning.pytorch as pl
from sklearn.metrics import roc_curve, auc

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.waveforms import SineGaussian, MultiSineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

from gwak.train.dataloader import SignalDataloader
from gwak.data.prior import SineGaussianBBC, MultiSineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from gwak.train.cl_models import Crayon

from gwak.train.plotting import make_corner

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

def frequency_cos_similarity(batch):
    H = torch.fft.rfft(batch[:, 0, :], dim=-1)
    L = torch.fft.rfft(batch[:, 1, :], dim=-1)
    numerator = torch.sum(H * torch.conj(L), dim=-1)
    norm_H = torch.linalg.norm(H, dim=-1)
    norm_L = torch.linalg.norm(L, dim=-1)
    rho_complex = numerator / (norm_H * norm_L + 1e-8)
    rho_real = torch.real(rho_complex).unsqueeze(-1)
    return rho_real

if __name__=='__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Process and merge ROOT files into datasets.')
    parser.add_argument('--embedding-model', type=str, default=None)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--ifos', type=str)
    parser.add_argument('--embeddings', type=str)
    parser.add_argument('--labels', type=str)
    parser.add_argument('--correlations', type=str)
    parser.add_argument('--means', type=str, default=None)
    parser.add_argument('--stds', type=str, default=None)
    parser.add_argument('--nevents', type=int, default=10000)
    args = parser.parse_args()

    embed_model = torch.jit.load(args.embedding_model)
    embed_model.eval()
    embed_model.to(device=device)

    # Load the YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract values
    sample_rate = config['data']['init_args']['sample_rate']
    kernel_length = config['data']['init_args']['kernel_length']
    psd_length = config['data']['init_args']['psd_length']
    fduration = config['data']['init_args']['fduration']
    fftlength = config['data']['init_args']['fftlength']
    batch_size = 128 #config['data']['init_args']['batch_size']
    batches_per_epoch = 2000000
    num_workers = config['data']['init_args']['num_workers']
    data_saving_file = config['data']['init_args']['data_saving_file']

    # Computed variable
    duration = fduration + kernel_length

    # Signal setup
    signal_classes = ['Glitch', 'Background']
    priors = [None, None]
    waveforms = [None, None]
    extra_kwargs = [None, None]

    # DataLoader
    loader = SignalDataloader(
        signal_classes, priors, waveforms, extra_kwargs,
        data_dir=args.data_dir,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        psd_length=psd_length,
        fduration=fduration,
        fftlength=fftlength,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        num_workers=num_workers,
        data_saving_file=data_saving_file,
        ifos=args.ifos,
        glitch_root=f'/home/hongyin.chen/anti_gravity/gwak/gwak/output/omicron/{args.ifos}/'
    )

    all_labels = []
    all_embeddings = []
    all_correlations = []

    n_iter = args.nevents // batch_size
    train_loader = loader.train_dataloader()
    train_iter = iter(train_loader)
    for i in range(n_iter):

        if i % 10 == 0:
            print(f"Processed batch {i}/{n_iter}")

        clean_batch, glitch_batch = next(train_iter)
        clean_batch = clean_batch.to(device)
        glitch_batch = glitch_batch.to(device)

        processed, labels, _ = loader.on_after_batch_transfer([clean_batch, glitch_batch], None, local_test=True)

        embeddings = embed_model(processed).cpu().detach().numpy()
        correlations = frequency_cos_similarity(processed).cpu().detach().numpy()

        all_labels.append(labels.cpu().detach().numpy())
        all_embeddings.append(embeddings)
        all_correlations.append(correlations)

        del clean_batch, glitch_batch, processed, embeddings, correlations
        torch.cuda.empty_cache()

    all_labels = np.concatenate(all_labels, axis=0)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_correlations = np.concatenate(all_correlations, axis=0)

    np.save(f'{args.labels}', all_labels)
    print('Labels shape', all_labels.shape)

    np.save(f'{args.embeddings}', all_embeddings)
    print('Embeddings shape', all_embeddings.shape)

    np.save(f'{args.correlations}', all_correlations)
    print('Correlation shape', all_correlations.shape)

    means = np.mean(all_embeddings, axis=0)
    stds = np.std(all_embeddings, axis=0)
    np.save(f'{args.means}', means)
    np.save(f'{args.stds}', stds)
