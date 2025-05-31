import os
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

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
    parser.add_argument('--embedding-model', type=str)
    parser.add_argument('--fm-model', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--nevents', type=int)
    parser.add_argument('--ifos', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--threshold-1yr', type=float)
    parser.add_argument('--conditioning', type=str2bool, default=False)
    args = parser.parse_args()

    embed_model = torch.jit.load(args.embedding_model)
    embed_model.eval()
    embed_model.to(device=device)

    # Load metric model
    metric_model = torch.jit.load(args.fm_model)
    metric_model.eval()
    metric_model.to(device=device)

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
    signal_classes = [
        "MultiSineGaussian",
        "SineGaussian",
        "BBH",
        "Gaussian",
        "Cusp",
        "Kink",
        "KinkKink",
        "WhiteNoiseBurst",
        "CCSN",
        "Background",
        "Glitch",
        "FakeGlitch"
        ]

    # Computed variable
    duration = fduration + kernel_length

    # Signal setup
    priors = [
        MultiSineGaussianBBC(), SineGaussianBBC(), LAL_BBHPrior(), GaussianBBC(),
        CuspBBC(), KinkBBC(), KinkkinkBBC(), WhiteNoiseBurstBBC(),
        None, None, None, None
    ]
    waveforms = [
        MultiSineGaussian(sample_rate=sample_rate, duration=duration),
        SineGaussian(sample_rate=sample_rate, duration=duration),
        IMRPhenomPv2(),
        Gaussian(sample_rate=sample_rate, duration=duration),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        WhiteNoiseBurst(sample_rate=sample_rate, duration=duration),
        None, None, None, None
    ]
    extra_kwargs = [
        None, None, {"ringdown_duration": 0.9}, None, None, None, None, None,
        None, None, None, None
    ]

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
        glitch_root=f'/home/hongyin.chen/anti_gravity/gwak/gwak/output/omicron/{args.ifos}/'
    )

    all_background_classes = ['Background', 'Glitch'] #, 'FakeGlitch']
    all_classes = signal_classes
    background_classes = [cls for cls in all_classes if cls in all_background_classes]
    signal_classes = [cls for cls in all_classes if cls not in background_classes]
    background_labels = [i+1 for i in range(len(all_classes)) if all_classes[i] in background_classes]
    signal_labels = [i+1 for i in range(len(all_classes)) if all_classes[i] in signal_classes]
    print('The signal classes are ', signal_classes)
    print('The background classes are ', background_classes)
    print('and the background labels are ', background_labels)

    all_binary_labels = []
    all_scores = []
    all_embeddings = []
    all_labels = []
    all_context = []
    all_snrs = []

    n_iter = int(args.nevents/batch_size)
    test_loader = loader.test_dataloader()
    test_iter = iter(test_loader)
    for i in tqdm(range(n_iter), desc="Processing batches"):
        clean_batch, glitch_batch = next(test_iter)
        clean_batch = clean_batch.to(device)
        glitch_batch = glitch_batch.to(device)

        processed, labels, snrs = loader.on_after_batch_transfer([clean_batch, glitch_batch], None,
            local_test=True)

        embeddings = embed_model(processed)

        if args.conditioning:
            context = frequency_cos_similarity(processed)
            all_context.append(context.detach().cpu().numpy())
            scores = metric_model(embeddings, context=context).detach().cpu().numpy() * (-1)
        else:
            scores = metric_model(embeddings).detach().cpu().numpy() * (-1)
        embeddings = embeddings.detach().cpu().numpy()

        labels = labels.detach().cpu().numpy()
        binary_labels = (~np.isin(labels, background_labels)).astype(int)


        all_binary_labels.append(binary_labels)
        all_labels.append(labels)
        all_scores.append(scores)
        all_embeddings.append(embeddings)
        all_snrs.append(snrs.detach().cpu().numpy())

        del clean_batch, glitch_batch, processed, embeddings, binary_labels, labels, scores, snrs
        torch.cuda.empty_cache()

    if args.conditioning:
        all_context = np.concatenate(all_context, axis=0)
    all_binary_labels = np.concatenate(all_binary_labels, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_snrs = np.concatenate(all_snrs, axis=0)

    np.save(f'{args.output}/context_{args.nevents}.npy', all_context)
    np.save(f'{args.output}/binary_labels_{args.nevents}.npy', all_binary_labels)
    np.save(f'{args.output}/labels_{args.nevents}.npy', all_labels)
    np.save(f'{args.output}/scores_{args.nevents}.npy', all_scores)
    np.save(f'{args.output}/embeddings_{args.nevents}.npy', all_embeddings)
    np.save(f'{args.output}/snrs_{args.nevents}.npy', all_snrs)

    # # Create mask where SNR > 4
    # snr_mask = all_snrs > 4
    # print('Passed hte SNR cut', 100 * np.mean(all_snrs > 4))
    # # Apply mask
    # all_context = all_context[snr_mask]
    # all_binary_labels = all_binary_labels[snr_mask]
    # all_labels = all_labels[snr_mask]
    # all_scores = all_scores[snr_mask]
    # all_embeddings = all_embeddings[snr_mask]
    # all_snrs = all_snrs[snr_mask]

    ########### PLOT CONTEXT
    # Mask based on label
    bkg_mask = all_binary_labels == 0
    sig_mask = all_binary_labels == 1

    if args.conditioning:
        # Plot histograms
        plt.figure()
        plt.hist(all_context[bkg_mask], bins=100, alpha=0.6, label="background", density=True)
        plt.hist(all_context[sig_mask], bins=100, alpha=0.6, label="signal", density=True)
        plt.xlabel("context")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Histogram of Context Values")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{args.output}/context.png')
        plt.clf()

    # Compute the ROC curve and AUC using scikit-learn
    fpr, tpr, thresholds = roc_curve(all_binary_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (All Signals)')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/roc_combined.png')
    plt.clf()

    # Compute and plot ROC curves for each anomaly class (all but "Background").
    plt.figure(figsize=(8,6))
    for i, anomaly_class in enumerate(signal_classes):
        # The anomaly class numeric label is i+1.
        anomaly_val = i + 1
        # Filter to only examples that are either the current anomaly class or Background (label 8).
        idx = np.where(np.isin(all_labels, background_labels + [anomaly_val]))[0]
        if idx.size == 0:
            continue  # skip if no examples
        scores_i = all_scores[idx]
        # Create binary ground truth: positive if the example is of the current anomaly class, 0 if background.
        binary_labels = (all_labels[idx] == anomaly_val).astype(int)

        # Compute the ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(binary_labels, scores_i)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr,
            lw=2,
            label=f'{anomaly_class} (AUC = {roc_auc:.2f})')

    # Plot reference line (diagonal)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Anomaly Class')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/rocs_bySignal.png')

    # ###################
    # ## Make corner plot
    fig = make_corner(all_embeddings, (all_labels-1).astype(int), return_fig=True, label_names=all_classes)
    fig.savefig(f'{args.output}/corner_plot.png')

    # Define custom colors
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939',
        '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#9c9ede', '#637939',
        '#e7ba52', '#ad494a'
    ]

    #####
    # Stack histograms properly
    plt.figure()

    # Collect all scores in a list
    score_list = []
    for i, c in enumerate(all_classes):
        scores_sel = all_scores[all_labels == i + 1]
        score_list.append(scores_sel)

    # Plot all at once
    plt.hist(
        score_list,
        bins=100,
        label=all_classes,
        alpha=0.8,
        range=(0, 500),
        stacked=True,
        color=custom_colors[:len(all_classes)]  # trim color list if needed
    )

    plt.xlabel("NF log probability")
    plt.legend()
    plt.savefig(f'{args.output}/metric.png')
    plt.clf()

    # -----------------------------------------------------------------------------
    # 3) BIN ANOMALIES BY SNR AND COMPUTE FRACTION DETECTED (SCORE > threshold_1yr)
    # -----------------------------------------------------------------------------
    # We'll define 10 bins in SNR from the minimum anomaly SNR to the maximum
    # anomaly SNR across all anomaly classes (labels 1..7).

    threshold_1yr = args.threshold_1yr

    anom_mask = ~np.isin(all_labels, background_labels)
    if np.any(anom_mask):
        snr_min, snr_max = all_snrs[anom_mask].min(), all_snrs[anom_mask].max()
    else:
        raise ValueError("No anomaly samples found in the test set!")

    num_bins = 10
    bin_edges = np.logspace(-2, 3, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8,6))

    for i, anom_class_name in enumerate(signal_classes):
        class_label = i + 1  # classes are 1..7
        mask = (all_labels == class_label)
        if not np.any(mask):
            # If no samples for that class, skip
            continue

        class_scores = all_scores[mask]
        class_snrs = all_snrs[mask]

        # Bin by SNR
        bin_idx = np.digitize(class_snrs, bin_edges) - 1  # bin indices in [0..num_bins-1]

        frac_detected = []
        for b in range(num_bins):
            in_bin = (bin_idx == b)
            if not np.any(in_bin):
                frac_detected.append(np.nan)  # or 0.0 if you prefer
            else:
                # fraction that exceed threshold
                frac = np.mean(class_scores[in_bin] > threshold_1yr)
                frac_detected.append(frac)

        plt.plot(bin_centers, frac_detected, marker='o', label=anom_class_name)

    plt.xlabel("SNR")
    plt.ylabel("Fraction of events detected")
    plt.title("Fraction of Events Detected at 1/Year FAR vs. SNR")
    plt.ylim([0, 1.05])
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{args.output}/fraction_1overYearFAR_SNR.png')

    bins = np.logspace(1,7,100)
    for i,sig in enumerate(signal_classes):
        h=plt.hist(all_snrs[all_labels==i+1],bins=np.logspace(-3,3,100),label=sig,histtype='step')
    plt.xscale('log')
    plt.xlabel("SNR")
    plt.legend()
    plt.savefig(f'{args.output}/snr_histograms.png')
    plt.clf()