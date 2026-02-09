import argparse
import numpy as np
import sys
from dataloader import SignalDataloader
import torch

from gwak.data.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

from tqdm import tqdm
import random
import h5py
import os

def main(ifos, num_samples_per_class, dataset):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = f"/home/katya.govorkova/gwak2/gwak/output/BBC_AnalysisReady_Cat12/{ifos}/test/"
    sample_rate = 4096
    kernel_length = 1.0
    psd_length = 64
    fduration = 2
    fftlength = 2
    batch_size = 128
    num_workers = 1
    duration = fduration + kernel_length

    random_id = random.randint(1000, 9999)
    OUTFILE = f"output/dataset_{dataset}_{ifos}_SR{sample_rate}_kernel{kernel_length}_{random_id}_with_GPS.h5"
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    signal_classes = ["SineGaussian",
                      "BBH",
                      # "Gaussian",
                      # "Cusp",
                      # "Kink",
                      # "KinkKink",
                      "WhiteNoiseBurst",
                      # "CCSN",
                      "Background",
                      "Glitch"]
    priors = [
        SineGaussianBBC(),
        LAL_BBHPrior(),
        # GaussianBBC(),
        # CuspBBC(),
        # KinkBBC(),
        # KinkkinkBBC(),
        WhiteNoiseBurstBBC(),
        # None,
        None,
        None
    ]
    waveforms = [
        SineGaussian(sample_rate=sample_rate, duration=duration),
        IMRPhenomPv2(),
        # Gaussian(sample_rate=sample_rate, duration=duration),
        # GenerateString(sample_rate=sample_rate),
        # GenerateString(sample_rate=sample_rate),
        # GenerateString(sample_rate=sample_rate),
        WhiteNoiseBurst(sample_rate=sample_rate, duration=duration),
        # None,
        None,
        None
    ]

    extra_kwargs = [
        None,
        {"ringdown_duration":0.9},
        # None,
        # None,
        # None,
        # None,
        None,
        # None,
        None,
        None
    ]
    snr_prior = torch.distributions.Uniform(3,30)

    # compute number of batches needed to get enough samples (equal number per class per batch)
    num_classes = len(signal_classes)
    num_samples = num_classes * num_samples_per_class
    batches_per_epoch = num_samples // batch_size + 1

    def write_file(filename, data, labels, snrs, clean, gps):
        with h5py.File(filename, "a") as f:
            uniq_labels = sorted(list(set(labels)))
            for l in uniq_labels:
                sig_name = signal_classes[int(l-1)]
                indices = labels == l
                class_data = data[indices]
                class_labels = labels[indices]
                class_snrs = snrs[indices]
                class_clean = clean[indices]
                class_gps = gps[indices]  # <-- GPS aligned with labels

                if f"{sig_name}_data" in f.keys():
                    # Append to existing datasets
                    f[f"{sig_name}_data"].resize(f[f"{sig_name}_data"].shape[0] + class_data.shape[0], axis=0)
                    f[f"{sig_name}_data"][-class_data.shape[0]:] = class_data

                    f[f"{sig_name}_snrs"].resize(f[f"{sig_name}_snrs"].shape[0] + class_snrs.shape[0], axis=0)
                    f[f"{sig_name}_snrs"][-class_snrs.shape[0]:] = class_snrs

                    # Append clean + gps only for Background
                    if sig_name == "Background":
                        f[f"{sig_name}_clean"].resize(f[f"{sig_name}_clean"].shape[0] + class_clean.shape[0], axis=0)
                        f[f"{sig_name}_clean"][-class_clean.shape[0]:] = class_clean

                        # Background_gps can be 1D or 2D depending on num_channels
                        if f"{sig_name}_gps" not in f.keys():
                            # create with correct dimensionality
                            if class_gps.ndim == 1:
                                f.create_dataset(f"{sig_name}_gps", data=class_gps.astype(np.float64),
                                                 maxshape=(None,), chunks=True)
                            else:  # (N, C)
                                f.create_dataset(f"{sig_name}_gps", data=class_gps.astype(np.float64),
                                                 maxshape=(None, class_gps.shape[1]), chunks=True)
                        else:
                            # append
                            f[f"{sig_name}_gps"].resize(f[f"{sig_name}_gps"].shape[0] + class_gps.shape[0], axis=0)
                            f[f"{sig_name}_gps"][-class_gps.shape[0]:] = class_gps.astype(np.float64)
                else:
                    # Create new datasets
                    f.create_dataset(f"{sig_name}_data", data=class_data,
                                     maxshape=(None, class_data.shape[1], class_data.shape[2]), chunks=True)
                    f.create_dataset(f"{sig_name}_snrs", data=class_snrs,
                                     maxshape=(None,), chunks=True)
                    # Create clean + gps only for Background
                    if sig_name == "Background":
                        f.create_dataset(f"{sig_name}_clean", data=class_clean,
                                         maxshape=(None, class_clean.shape[1], class_clean.shape[2]), chunks=True)
                        if class_gps.ndim == 1:
                            f.create_dataset(f"{sig_name}_gps", data=class_gps.astype(np.float64),
                                             maxshape=(None,), chunks=True)
                        else:
                            f.create_dataset(f"{sig_name}_gps", data=class_gps.astype(np.float64),
                                             maxshape=(None, class_gps.shape[1]), chunks=True)
        # ------------------------------------------------------------------

    sig_loader = SignalDataloader(signal_classes,
        priors,
        waveforms,
        extra_kwargs,
        data_dir=data_dir,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        psd_length=psd_length,
        fduration=fduration,
        fftlength=fftlength,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        num_workers=num_workers,
        ifos=ifos,
        glitch_root=f"/home/hongyin.chen/anti_gravity/gwak/gwak/output/O4b_AnalysisReady_Cat12/omicron/",
        # remake_cache=True,
        anneal_snr=False,
        snr_prior=snr_prior,
        rebalance_classes=False
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
    clean = []
    glitch = []  # <-- only change: collect glitch batches
    num_loaded = 0
    max_in_memory = 5_000
    gps = []  # <-- add near the other accumulators

    for i in tqdm(range(num_batches)):
        # If your loader returns gps with clean_batch, unpack it here.
        # Example A: loader yields (clean_wave, glitch_wave) and you compute t_gps from sig_loader:
        #   clean_batch, glitch_batch = next(data_iter)
        #   ...
        #   t_gps = sig_loader.last_gps  # (example placeholder)
        #
        # Example B (common after your earlier change): loader yields ((clean_wave, clean_gps), (glitch_wave, glitch_gps))
        # Unpack like this:
        (clean_wave, clean_tgps), (glitch_wave, glitch_tgps) = next(data_iter)

        clean_batch = clean_wave.to(device)
        glitch_batch = glitch_wave.to(device)

        batch, indexed_labels, snr = sig_loader.on_after_batch_transfer([clean_batch, glitch_batch], None, local_test=True)

        data.append(batch.cpu().numpy().astype(np.float32))
        labels.append(indexed_labels.cpu().numpy().astype(np.int32))
        snrs.append(snr.cpu().numpy().astype(np.float32))
        clean.append(clean_batch.cpu().numpy().astype(np.float32))

        # choose which GPS aligns with the final 'batch' entries:
        # If 'batch' is built from clean windows, use clean_tgps:
        gps.append(np.asarray(clean_tgps))   # ensure shape (N,) or (N, C), dtype float64 preferred

        num_loaded += batch.shape[0]
        if num_loaded >= max_in_memory:
            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
            snrs = np.concatenate(snrs, axis=0)
            clean = np.concatenate(clean, axis=0)
            gps = np.concatenate(gps, axis=0)
            write_file(OUTFILE, data, labels, snrs, clean, gps)

            # clean up
            del data, labels, snrs, clean, gps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            data, labels, snrs, clean, gps = [], [], [], [], []
            num_loaded = 0

    if num_loaded > 0:
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        snrs = np.concatenate(snrs, axis=0)
        clean = np.concatenate(clean, axis=0)
        gps = np.concatenate(gps, axis=0)
        write_file(OUTFILE, data, labels, snrs, clean, gps)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create offline dataset.')
    parser.add_argument('ifos', type=str, help='(HL,HV,LV)')
    parser.add_argument('num_samples', type=int, help='per class')
    parser.add_argument('dataset', type=str, help='(train,val,test)')
    args = parser.parse_args()

    main(args.ifos, args.num_samples, args.dataset)