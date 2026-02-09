# import numpy as np
# import sys

# from gwak.train.dataloader import SignalDataloader
# import torch

# from gwak.data.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
# from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst
# import matplotlib.pyplot as plt
# import sys
# import random
# import h5py
# from tqdm import tqdm

# # assert len(sys.argv) == 3
# n_events = 10000
# out_dir = '/home/katya.govorkova/gwak2/gwak/output/for_Nastya'

# random_id = random.randint(1000, 9999)
# out_file = f"{out_dir}/offline_samples_{random_id}.h5"

# data_dir = "/home/katya.govorkova/gwak2/gwak/output/O3a"
# sample_rate = 4096
# kernel_length = 1.0
# psd_length = 64
# fduration = 2
# fftlength = 2
# batch_size = 128
# batches_per_epoch = n_events // batch_size + 1
# num_workers = 4
# duration = fduration + kernel_length

# signal_classes = ["SineGaussian",
#                   "BBH",
#                   "Gaussian",
#                   "Cusp",
#                   "Kink",
#                   "KinkKink",
#                   "WhiteNoiseBurst",
#                   #"CCSN",
#                   #"Background",
#                   #"Glitch"
# ]
# priors = [
#     SineGaussianBBC(),
#     LAL_BBHPrior(),
#     GaussianBBC(),
#     CuspBBC(),
#     KinkBBC(),
#     KinkkinkBBC(),
#     WhiteNoiseBurstBBC(),
#     #None,
#     #None,
#     #None
# ]
# waveforms = [
#     SineGaussian(sample_rate=sample_rate, duration=duration),
#     IMRPhenomPv2(),
#     Gaussian(sample_rate=sample_rate, duration=duration),
#     GenerateString(sample_rate=sample_rate),
#     GenerateString(sample_rate=sample_rate),
#     GenerateString(sample_rate=sample_rate),
#     WhiteNoiseBurst(sample_rate=sample_rate, duration=duration),
#     #None,
#     #None,
#     #None
# ]

# extra_kwargs = [
#     None,
#     {"ringdown_duration":0.9},
#     None,
#     None,
#     None,
#     None,
#     None,
#     #None,
#     #None,
#     #None
# ]

# snr_prior = torch.distributions.Uniform(3,50)

# loader = SignalDataloader(
#     signal_classes,
#     priors,
#     waveforms,
#     extra_kwargs,
#     data_dir=data_dir,
#     sample_rate=sample_rate,
#     kernel_length=kernel_length,
#     psd_length=psd_length,
#     fduration=fduration,
#     fftlength=fftlength,
#     batch_size=batch_size,
#     batches_per_epoch=batches_per_epoch,
#     num_workers=num_workers,
#     ifos="HL",
#     loader_mode="raw",
#     snr_prior=snr_prior
# )

# train_loader = loader.train_dataloader()

# def write_to_file(bkg_arr, waveforms_arr, labels_arr, snrs_arr):
#     with h5py.File(out_file,"a") as f:
#         if "bkg" in f.keys():
#             f["bkg"].resize(f["bkg"].shape[0] + bkg_arr.shape[0], axis=0)
#             f["bkg"][-bkg_arr.shape[0]:] = bkg_arr

#             f["sig_waveforms"].resize(f["sig_waveforms"].shape[0] + waveforms_arr.shape[0], axis=0)
#             f["sig_waveforms"][-waveforms_arr.shape[0]:] = waveforms_arr

#             f["sig_labels"].resize(f["sig_labels"].shape[0] + labels_arr.shape[0], axis=0)
#             f["sig_labels"][-labels_arr.shape[0]:] = labels_arr

#             f["sig_snrs"].resize(f["sig_snrs"].shape[0] + snrs_arr.shape[0], axis=0)
#             f["sig_snrs"][-snrs_arr.shape[0]:] = snrs_arr
#         else:
#             f.create_dataset("bkg", data=bkg_arr,
#                                 maxshape=(None,bkg_arr.shape[1],bkg_arr.shape[2]), chunks=True)

#             f.create_dataset("sig_waveforms", data=waveforms_arr,
#                                 maxshape=(None,waveforms_arr.shape[1],waveforms_arr.shape[2]), chunks=True)

#             f.create_dataset("sig_labels", data=labels_arr,
#                                 maxshape=(None,), chunks=True)

#             f.create_dataset("sig_snrs", data=snrs_arr,
#                                 maxshape=(None,), chunks=True)

# EVENT_BUFFER = 5_000

# event_count = 0
# all_bkg = []
# all_waveforms = []
# all_labels = []
# all_snrs = []

# for batch in tqdm(train_loader):
#     clean_batch, waveforms, labels, indexed_labels, snrs = loader.on_after_batch_transfer(batch,None)
#     event_count += clean_batch.shape[0]
#     all_bkg.append(clean_batch.cpu().numpy())
#     all_waveforms.append(waveforms.cpu().numpy())
#     all_labels.append(labels.cpu().numpy())
#     all_snrs.append(snrs.cpu().numpy())

#     if event_count >= EVENT_BUFFER:
#         all_bkg = np.concatenate(all_bkg, axis=0)
#         all_waveforms = np.concatenate(all_waveforms, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
#         all_snrs = np.concatenate(all_snrs, axis=0)

#         write_to_file(all_bkg, all_waveforms, all_labels, all_snrs)

#         all_bkg = []
#         all_waveforms = []
#         all_labels = []
#         all_snrs = []
#         event_count = 0
# if event_count > 0:
#     write_to_file(
#         np.concatenate(all_bkg, axis=0),
#         np.concatenate(all_waveforms, axis=0),
#         np.concatenate(all_labels, axis=0),
#         np.concatenate(all_snrs, axis=0)
#     )

# signal_labels = []
# for c in loader.signal_classes:
#     c_label = loader.all_signal_labels[c]
#     if c_label in signal_labels:
#         continue
#     else:
#         signal_labels.append(c_label)
# signal_classes = [loader.all_signal_label_names[c] for c in signal_labels]
# signal_classes = np.array(signal_classes,dtype=object)
# str_dtype = h5py.string_dtype(encoding='utf-8')

# with h5py.File(out_file,"a") as f:
#     f.create_dataset("signal_classes", data=signal_classes,dtype=str_dtype)
#     f.create_dataset("signal_labels", data=np.array(signal_labels,dtype=np.int32))

# print("DONE")


import argparse
import numpy as np
import sys
from gwak.train.dataloader import SignalDataloader
import torch

from gwak.data.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

from tqdm import tqdm
import random
import h5py
import os

def main(ifos, num_samples_per_class, dataset):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = "/home/katya.govorkova/gwak2/gwak/output/O3a"
    sample_rate = 4096
    kernel_length = 1.0
    psd_length = 64
    fduration = 2
    fftlength = 2
    batch_size = 128
    num_workers = 1
    duration = fduration + kernel_length


    OUTFILE = f"/home/katya.govorkova/gwak2/gwak/output/for_Nastya/dataset.h5"
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    signal_classes = ["SineGaussian",
                    "BBH",
                    "Gaussian",
                    "Cusp",
                    "Kink",
                    "KinkKink",
                    "WhiteNoiseBurst",
                    # "CCSN",
                    "Background",
                    # "Glitch"
                    ]
    priors = [
        SineGaussianBBC(),
        LAL_BBHPrior(),
        GaussianBBC(),
        CuspBBC(),
        KinkBBC(),
        KinkkinkBBC(),
        WhiteNoiseBurstBBC(),
        # None,
        None,
        # None
    ]
    waveforms = [
        SineGaussian(sample_rate=sample_rate, duration=duration),
        IMRPhenomPv2(),
        Gaussian(sample_rate=sample_rate, duration=duration),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        WhiteNoiseBurst(sample_rate=sample_rate, duration=duration),
        # None,
        None,
        # None
    ]

    extra_kwargs = [
        None,
        {"ringdown_duration":0.9},
        None,
        None,
        None,
        None,
        None,
        # None,
        None,
        # None
    ]
    snr_prior = torch.distributions.Uniform(3,30)

    # compute number of batches needed to get enough samples (equal number per class per batch)
    num_classes = len(signal_classes)
    num_samples = num_classes * num_samples_per_class
    batches_per_epoch = num_samples // batch_size + 1

    def write_file(filename, data, labels, snrs):
        with h5py.File(filename, "a") as f:
            uniq_labels = sorted(list(set(labels)))
            for l in uniq_labels:
                sig_name = signal_classes[int(l-1)]
                indices = labels == l
                class_data = data[indices]
                class_labels = labels[indices]
                class_snrs = snrs[indices]
                # Check if datasets exist
                if f"{sig_name}_data" in f.keys():
                    # Append to existing datasets
                    f[f"{sig_name}_data"].resize(f[f"{sig_name}_data"].shape[0] + class_data.shape[0], axis=0)
                    f[f"{sig_name}_data"][-class_data.shape[0]:] = class_data

                    # f[f"{sig_name}_labels"].resize(f[f"{sig_name}_labels"].shape[0] + class_labels.shape[0], axis=0)
                    # f[f"{sig_name}_labels"][-class_labels.shape[0]:] = class_labels

                    f[f"{sig_name}_snrs"].resize(f[f"{sig_name}_snrs"].shape[0] + class_snrs.shape[0], axis=0)
                    f[f"{sig_name}_snrs"][-class_snrs.shape[0]:] = class_snrs
                else:
                    # Create new datasets with chunks for efficient appending later
                    f.create_dataset(f"{sig_name}_data", data=class_data,
                                        maxshape=(None,class_data.shape[1],class_data.shape[2]), chunks=True)
                    # f.create_dataset(f"{sig_name}_labels", data=class_labels,
                    #                    maxshape=(None,), chunks=True)
                    f.create_dataset(f"{sig_name}_snrs", data=class_snrs,
                                        maxshape=(None,), chunks=True)

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
        # loader_mode='raw',
        glitch_root='/home/katya.govorkova/gw_anomaly/output/omicron_for_dataloader', # f"/home/hongyin.chen/anti_gravity/gwak/gwak/output/O4b_AnalysisReady_Cat12/omicron/",
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
    num_loaded = 0
    max_in_memory = 5_000
    for i in tqdm(range(num_batches)):
        clean_batch, _ = next(data_iter)
        clean_batch = clean_batch.to(device)
        batch, indexed_labels, snr = sig_loader.on_after_batch_transfer([clean_batch, None],None,local_test=True)
        data.append(batch.cpu().numpy().astype(np.float32))
        labels.append(indexed_labels.cpu().numpy().astype(np.int32))
        snrs.append(snr.cpu().numpy().astype(np.float32))
        num_loaded += batch.shape[0]
        if num_loaded >= max_in_memory:
            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
            snrs = np.concatenate(snrs, axis=0)
            write_file(OUTFILE, data, labels, snrs)

            # clean up
            del data, labels, snrs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            data = []
            labels = []
            snrs = []
            num_loaded = 0
    if num_loaded > 0:
        print('Loaded data ', data)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        snrs = np.concatenate(snrs, axis=0)
        write_file(OUTFILE, data, labels, snrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create offline dataset.')
    parser.add_argument('ifos', type=str, help='(HL,HV,LV)')
    parser.add_argument('num_samples', type=int, help='per class')
    parser.add_argument('dataset', type=str, help='(train,val,test)')
    args = parser.parse_args()

    main(args.ifos, args.num_samples, args.dataset)

