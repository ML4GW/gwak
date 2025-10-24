import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from sklearn.metrics import roc_curve, auc

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.waveforms import SineGaussian, MultiSineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

from gwak.train.dataloader import SignalDataloader
from gwak.data.prior import SineGaussianBBC, MultiSineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from gwak.train.cl_models import Crayon

from gwak.train.plotting import make_corner

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

if __name__=='__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Process and merge ROOT files into datasets.')
    parser.add_argument('--embedding-model', type=str, default=None)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--means', type=str)
    parser.add_argument('--stds', type=str)
    args = parser.parse_args()

    import torch
    from pytorch_lightning import Trainer
    import yaml

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
    batch_size = 256 #config['data']['init_args']['batch_size']
    batches_per_epoch = config['data']['init_args']['batches_per_epoch']
    num_workers = config['data']['init_args']['num_workers']
    data_saving_file = config['data']['init_args']['data_saving_file']

    # Computed variable
    duration = fduration + kernel_length

    # Signal setup
    signal_classes = ['Glitch', 'FakeGlitch', 'Background']
    priors = [None, None, None]
    waveforms = [None, None, None]
    extra_kwargs = [None, None, None]

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
        # glitch_root='/home/hongyin.chen/anti_gravity/gwak/gwak/output/omicron/HL/'
        glitch_root=f"/fred/oz016/Andy/New_Data/gwak/omicron/HL"
    )

    all_embeddings = []

    n_iter = 200
    test_loader = loader.train_dataloader()
    test_iter = iter(test_loader)
    for i in range(n_iter):
        clean_batch, glitch_batch = next(test_iter)
        clean_batch = clean_batch.to(device)
        glitch_batch = glitch_batch.to(device)

        processed, labels = loader.on_after_batch_transfer([clean_batch, glitch_batch], None, local_test=True)

        embeddings = embed_model(processed).cpu().detach().numpy()
        all_embeddings.append(embeddings)

        del clean_batch, glitch_batch, processed, embeddings
        torch.cuda.empty_cache()

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    means = np.mean(all_embeddings, axis=0)
    stds = np.std(all_embeddings, axis=0)
    print(all_embeddings.shape)

    np.save(f'{args.means}', means)
    np.save(f'{args.stds}', stds)


