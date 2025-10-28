#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from dataloader import SignalDataloader
import torch

from gwak.data.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst
from gwak.train.preselection import cwb_stats_2ifo, cwb_cc_rho_max_over_delay_2ifo

from tqdm import tqdm
import random
import h5py
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # only old GPU1 is visible as cuda:0

def main(ifos, num_samples_per_class, dataset,
         flo=30.0, fhi=2048.0, sample_rate=4096.0, delay_scan=False, tau_max=0.010, n_tau=81,
         cc_min=None, rho_min=None, scc_min=None, edr_max=None, preselection=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = f"/home/katya.govorkova/gwak2/gwak/output/BBC_AnalysisReady_Cat12/{ifos}/"
    kernel_length = 1.0
    psd_length = 64
    fduration = 2
    fftlength = 3
    batch_size = 128
    num_workers = 1
    duration = fduration + kernel_length

    random_id = random.randint(1000, 9999)
    OUTFILE = f"output/dataset_{dataset}_{ifos}_SR{int(sample_rate)}_kernel{kernel_length}_{random_id}.h5"
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
        "Glitch"]
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
        SineGaussian(sample_rate=sample_rate, duration=duration),
        IMRPhenomPv2(),
        Gaussian(sample_rate=sample_rate, duration=duration),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        WhiteNoiseBurst(sample_rate=sample_rate, duration=duration),
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
    snr_prior = torch.distributions.Uniform(3,30)

    # compute number of batches needed to get enough samples (equal number per class per batch)
    num_classes = len(signal_classes)
    num_samples = num_classes * num_samples_per_class
    batches_per_epoch = num_samples // batch_size + 1

    def _passes_cuts(cls_name, cc, rho, scc, edr):
        if (cc_min is not None and cc < cc_min):   return False
        if (rho_min is not None and rho < rho_min): return False
        if (scc_min is not None and scc < scc_min): return False
        if (edr_max is not None and edr > edr_max): return False
        return True

    def write_file(filename, data, labels, snrs, ccs, rhos, sccs, edrs):
        with h5py.File(filename, "a") as f:
            uniq_labels = sorted(list(set(labels)))
            for l in uniq_labels:
                sig_name = signal_classes[int(l-1)]
                indices = labels == l
                class_data = data[indices]
                class_snrs = snrs[indices]

                # If stats are provided, slice them; otherwise keep as None
                class_cc  = ccs[indices]  if ccs  is not None else None
                class_rho = rhos[indices] if rhos is not None else None
                class_scc = sccs[indices] if sccs is not None else None
                class_edr = edrs[indices] if edrs is not None else None

                if f"{sig_name}_data" in f.keys():
                    # data
                    f[f"{sig_name}_data"].resize(f[f"{sig_name}_data"].shape[0] + class_data.shape[0], axis=0)
                    f[f"{sig_name}_data"][-class_data.shape[0]:] = class_data
                    # snr
                    f[f"{sig_name}_snrs"].resize(f[f"{sig_name}_snrs"].shape[0] + class_snrs.shape[0], axis=0)
                    f[f"{sig_name}_snrs"][-class_snrs.shape[0]:] = class_snrs

                    # stats (only if provided)
                    if class_cc is not None:
                        for name, arr in [("cc", class_cc), ("rho", class_rho), ("scc", class_scc), ("edr", class_edr)]:
                            ds = f[f"{sig_name}_{name}"]
                            ds.resize(ds.shape[0] + arr.shape[0], axis=0)
                            ds[-arr.shape[0]:] = arr
                else:
                    f.create_dataset(f"{sig_name}_data", data=class_data,
                                     maxshape=(None, class_data.shape[1], class_data.shape[2]), chunks=True)
                    f.create_dataset(f"{sig_name}_snrs", data=class_snrs, maxshape=(None,), chunks=True)

                    # stats datasets (only if provided)
                    if class_cc is not None:
                        f.create_dataset(f"{sig_name}_cc",  data=class_cc,  maxshape=(None,), chunks=True)
                        f.create_dataset(f"{sig_name}_rho", data=class_rho, maxshape=(None,), chunks=True)
                        f.create_dataset(f"{sig_name}_scc", data=class_scc, maxshape=(None,), chunks=True)
                        f.create_dataset(f"{sig_name}_edr", data=class_edr, maxshape=(None,), chunks=True)

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
        rebalance_classes=False,
        whiten=False
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
    ccs  = []
    rhos = []
    sccs = []
    edrs = []
    num_loaded = 0
    max_in_memory = 5_000

    for _ in tqdm(range(num_batches)):
        clean_batch, glitch_batch = next(data_iter)
        clean_batch = clean_batch.to(device)
        glitch_batch = glitch_batch.to(device)
        batch, indexed_labels, snr = sig_loader.on_after_batch_transfer([clean_batch,glitch_batch], None, local_test=True)
        # batch: (B, 2, T)
        batch_np = batch.cpu().numpy().astype(np.float32)
        labels_np = indexed_labels.cpu().numpy().astype(np.int32)
        snr_np = snr.cpu().numpy().astype(np.float32)

        B = batch_np.shape[0]

        if preselection:
            # === compute stats per sample, apply cuts class-wise ===
            cc_list, rho_list, scc_list, edr_list = [], [], [], []
            keep = np.ones(B, dtype=bool)
            for i in range(B):
                h = batch_np[i, 0, :]
                l = batch_np[i, 1, :]
                if delay_scan:
                    cc, rho, scc_v, edr = cwb_cc_rho_max_over_delay_2ifo(
                        h, l, sample_rate, flo, fhi,
                        max_tau=tau_max, n_tau=n_tau
                    )
                else:
                    st = cwb_stats_2ifo(
                        h, l, sample_rate, flo, fhi
                    )
                    cc, rho, scc_v, edr = st["cc"], st["rho"], st["scc"], st["edr"]
                cc_list.append(cc); rho_list.append(rho); scc_list.append(scc_v); edr_list.append(edr)

                cls_name = signal_classes[int(labels_np[i]-1)]
                if not _passes_cuts(cls_name, cc, rho, scc_v, edr):
                    keep[i] = False

            cc_arr  = np.asarray(cc_list,  dtype=np.float32)
            rho_arr = np.asarray(rho_list, dtype=np.float32)
            scc_arr = np.asarray(scc_list, dtype=np.float32)
            edr_arr = np.asarray(edr_list, dtype=np.float32)

            # keep only passing samples
            batch_np  = batch_np[keep]
            labels_np = labels_np[keep]
            snr_np    = snr_np[keep]
            cc_arr    = cc_arr[keep]
            rho_arr   = rho_arr[keep]
            scc_arr   = scc_arr[keep]
            edr_arr   = edr_arr[keep]

            if batch_np.shape[0] == 0:
                continue

            # accumulate in RAM
            ccs.append(cc_arr)
            rhos.append(rho_arr)
            sccs.append(scc_arr)
            edrs.append(edr_arr)
        else:
            # No preselection: keep everything; do not compute or store stats
            if batch_np.shape[0] == 0:
                continue

        data.append(batch_np)
        labels.append(labels_np)
        snrs.append(snr_np)

        num_loaded += batch_np.shape[0]
        if num_loaded >= max_in_memory:
            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
            snrs = np.concatenate(snrs, axis=0)

            if preselection:
                ccs_cat  = np.concatenate(ccs, axis=0)
                rhos_cat = np.concatenate(rhos, axis=0)
                sccs_cat = np.concatenate(sccs, axis=0)
                edrs_cat = np.concatenate(edrs, axis=0)
            else:
                ccs_cat = rhos_cat = sccs_cat = edrs_cat = None

            write_file(OUTFILE, data, labels, snrs, ccs_cat, rhos_cat, sccs_cat, edrs_cat)

            # clean up
            del data, labels, snrs
            if preselection:
                del ccs, rhos, sccs, edrs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            data, labels, snrs = [], [], []
            ccs, rhos, sccs, edrs = [], [], [], []
            num_loaded = 0

    if num_loaded > 0:
        data  = np.concatenate(data, axis=0)
        labels= np.concatenate(labels, axis=0)
        snrs  = np.concatenate(snrs, axis=0)

        if preselection and len(ccs) > 0:
            ccs_cat  = np.concatenate(ccs, axis=0)
            rhos_cat = np.concatenate(rhos, axis=0)
            sccs_cat = np.concatenate(sccs, axis=0)
            edrs_cat = np.concatenate(edrs, axis=0)
        else:
            ccs_cat = rhos_cat = sccs_cat = edrs_cat = None

        write_file(OUTFILE, data, labels, snrs, ccs_cat, rhos_cat, sccs_cat, edrs_cat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create offline dataset.')
    parser.add_argument('ifos', type=str, help='(HL,HV,LV)')
    parser.add_argument('num_samples', type=int, help='per class')
    parser.add_argument('dataset', type=str, help='(train,val,test)')

    # === NEW: toggle preselection ===
    parser.add_argument('--preselection', action='store_true', default=False,
                        help='If set, compute stats, apply cuts, and store cc/rho/scc/edr.')

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
        args.ifos, args.num_samples, args.dataset,
        flo=args.flo, fhi=args.fhi, sample_rate=args.fs,
        delay_scan=args.delay_scan, tau_max=args.tau_max, n_tau=args.n_tau,
        cc_min=args.cc_min, rho_min=args.rho_min,
        scc_min=args.scc_min, edr_max=args.edr_max,
        preselection=args.preselection,
    )