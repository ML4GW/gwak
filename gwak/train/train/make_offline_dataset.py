#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from dataloader import SignalDataloader
import torch

from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst
from ml4gw.distributions import PowerLaw

from trainer.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from pathlib import Path

from tqdm import tqdm

import random
import h5py
import os

from machinery import gwak_data_dir, gwak_dir
DATA_DIR = Path(os.getenv("GWAK_DATA_DIR"))

def write_file(filename, signal_classes, data, labels, snrs, hrss):
    with h5py.File(filename, "a") as f:
        for label in np.unique(labels):
            mask = labels == label
            group = f.require_group(signal_classes[int(label) - 1])

            for name, arr in [("data", data[mask]), ("snrs", snrs[mask]), ("hrss", hrss[mask])]:
                if name not in group:
                    group.create_dataset(name, data=arr, maxshape=(None, *arr.shape[1:]), chunks=True)
                else:
                    ds = group[name]
                    n = ds.shape[0]
                    ds.resize(n + len(arr), axis=0)
                    ds[n:] = arr

def main(
    ifos,
    num_samples_per_class,
    dataset,
    data_tag,
    flo=30.0,
    fhi=2048.0,
    sample_rate=4096.0,
    delay_scan=False,
    tau_max=0.010,
    n_tau=81,
    cc_min=None,
    rho_min=None,
    scc_min=None,
    edr_max=None,
    preselection=False
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _device = "cpu"

    # data_tag = "O4_MDC_background-chunked"
    data_dir = DATA_DIR / f"{data_tag}/{ifos}/"
    kernel_length = 1.0
    psd_length = 64
    fduration = 2
    fftlength = kernel_length + fduration # 3
    batch_size = 256
    num_workers = 4
    duration = fduration + kernel_length

    OUTFILE = DATA_DIR / f"dataset_{dataset}_{ifos}_SR{int(sample_rate)}_kernel{kernel_length}_hrss.h5"

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    signal_classes = [
        "SineGaussian",
        "BBH",
        "Gaussian",
        "Cusp",
        "Kink",
        "KinkKink",
        "WhiteNoiseBurst",
        "CCSN",
        "Background",
        "Glitch"
    ]

    priors = [
        SineGaussianBBC(),
        LAL_BBHPrior(),
        GaussianBBC(),
        CuspBBC(),
        KinkBBC(),
        KinkkinkBBC(),
        WhiteNoiseBurstBBC(),
        None,
        None,
        None
    ]
    waveforms = [
        SineGaussian(sample_rate=sample_rate, duration=duration).to(_device),
        IMRPhenomPv2().to(_device),
        Gaussian(sample_rate=sample_rate, duration=duration).to(_device),
        GenerateString(sample_rate=sample_rate, duration=duration).to(_device),
        GenerateString(sample_rate=sample_rate, duration=duration).to(_device),
        GenerateString(sample_rate=sample_rate, duration=duration).to(_device),
        WhiteNoiseBurst(sample_rate=sample_rate, duration=duration).to(_device),
        None,
        None,
        None
    ]

    extra_kwargs = [
        None,
        {"ringdown_duration":0.9},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    ]
    snr_prior = PowerLaw(index=-3, minimum=4, maximum=50)

    # compute number of batches needed to get enough samples (equal number per class per batch)
    num_classes = len(signal_classes)
    num_samples = num_classes * num_samples_per_class
    batches_per_epoch = num_samples // batch_size + 1


    sig_loader = SignalDataloader(
        signal_classes,
        priors,
        waveforms,
        extra_kwargs,
        data_tag=data_tag,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        psd_length=psd_length,
        fduration=fduration,
        fftlength=fftlength,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        num_workers=num_workers,
        ifos=ifos,
        glitch_root=DATA_DIR / f"O4b_AnalysisReady_Cat12/omicron/",
        # remake_cache=True,
        anneal_snr=False,
        snr_prior=snr_prior,
        rebalance_classes=False,
        whiten=True
    )

    if dataset == "train":
        loader = sig_loader.train_dataloader()
    elif dataset == "val":
        loader = sig_loader.val_dataloader()
    elif dataset == "test":
        loader = sig_loader.test_dataloader()
    data_iter = iter(loader)

    num_batches = batches_per_epoch
    data = []
    labels = []
    snrs = []
    hrsss = []

    num_loaded = 0
    max_in_memory = 5_000
    # breakpoint()
    for _ in tqdm(range(num_batches)):
        clean_batch, glitch_batch = next(data_iter)
        clean_batch = clean_batch.to(device)
        glitch_batch = glitch_batch.to(device)
        batch, indexed_labels, snr, hrss = sig_loader.on_after_batch_transfer(
            [clean_batch, glitch_batch],
            None,
            local_test=True
        )
        # batch: (B, 2, T)
        batch_np = batch.cpu().numpy().astype(np.float32)
        labels_np = indexed_labels.cpu().numpy().astype(np.int32)
        snr_np = snr.cpu().numpy().astype(np.float32)

        if batch_np.shape[0] == 0:
            continue

        data.append(batch_np)
        labels.append(labels_np)
        snrs.append(snr_np)
        hrsss.append(hrss.cpu().numpy().astype(np.float32))

        num_loaded += batch_np.shape[0]

        if num_loaded >= max_in_memory:

            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
            snrs = np.concatenate(snrs, axis=0)
            hrsss = np.concatenate(hrsss, axis=0)

            write_file(
                OUTFILE, signal_classes, data, labels, snrs, hrsss,
            )

            # clean up
            del data, labels, snrs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            data = []
            labels = []
            snrs = []
            hrsss = []
            num_loaded = 0

    if num_loaded > 0:
        print('Loaded data ', data)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        snrs = np.concatenate(snrs, axis=0)
        hrsss = np.concatenate(hrsss, axis=0)

        write_file(
            OUTFILE, signal_classes, data, labels, snrs, hrsss,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create offline dataset.')
    parser.add_argument('ifos', type=str, help='(HL,HV,LV)')
    parser.add_argument('num_samples', type=int, help='per class')
    parser.add_argument('dataset', type=str, help='(train,val,test)')
    parser.add_argument('data_tag', type=str, help='(data_version)')

    parser.add_argument(
        '--preselection', action='store_true', default=False,
        help='If set, compute stats, apply cuts, and store cc/rho/scc/edr.'
    )

    # === NEW: stats config & cuts (signals and background separately) ===
    parser.add_argument('--flo', type=float, default=30.0)
    parser.add_argument('--fhi', type=float, default=2048.0)
    parser.add_argument('--fs',  type=float, default=4096.0)
    parser.add_argument('--delay-scan', action='store_true', default=True)
    parser.add_argument('--tau-max', type=float, default=0.010)
    parser.add_argument('--n-tau', type=int, default=81)

    # background cuts
    parser.add_argument('--cc-min',  type=float, default=0.52626751)
    parser.add_argument('--rho-min', type=float, default=1.01491416)
    parser.add_argument('--scc-min', type=float, default=0.00024060)
    parser.add_argument('--edr-max', type=float, default=0.90017429)

    args = parser.parse_args()
    main(
        args.ifos, args.num_samples, args.dataset, args.data_tag,
        flo=args.flo, fhi=args.fhi, sample_rate=args.fs,
        delay_scan=args.delay_scan, tau_max=args.tau_max, n_tau=args.n_tau,
        cc_min=args.cc_min, rho_min=args.rho_min,
        scc_min=args.scc_min, edr_max=args.edr_max,
        preselection=args.preselection,
    )