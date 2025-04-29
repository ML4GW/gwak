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
from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

from gwak.train.dataloader import SignalDataloader
from gwak.data.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from gwak.train.cl_models import Crayon

device = torch.device('cuda:1') if torch.cuda.is_available() else 'cpu'


if __name__=='__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Process and merge ROOT files into datasets.')
    parser.add_argument('--embedding-model', type=str, default=None)
    parser.add_argument('--fm-model', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--use-freq-correlation', action='store_true',
        help='Include frequency-domain correlation as an additional feature')
    args = parser.parse_args()

    # Load the YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract values
    sample_rate = config['data']['init_args']['sample_rate']
    kernel_length = config['data']['init_args']['kernel_length']
    psd_length = config['data']['init_args']['psd_length']
    fduration = config['data']['init_args']['fduration']
    fftlength = config['data']['init_args']['fftlength']
    batch_size = 512 # config['data']['init_args']['batch_size']
    batches_per_epoch = config['data']['init_args']['batches_per_epoch']
    num_workers = config['data']['init_args']['num_workers']
    data_saving_file = config['data']['init_args']['data_saving_file']

    # Computed variable
    duration = fduration + kernel_length

    signal_classes = config['data']['init_args']['signal_classes']
    priors = [
        SineGaussianBBC(),
        LAL_BBHPrior(),
        GaussianBBC(),
        CuspBBC(),
        KinkBBC(),
        KinkkinkBBC(),
        WhiteNoiseBurstBBC(),
        None
    ]
    waveforms = [
        SineGaussian(
            sample_rate=sample_rate,
            duration=duration
        ),
        IMRPhenomPv2(),
        Gaussian(
            sample_rate=sample_rate,
            duration=duration
        ),
        GenerateString(
            sample_rate=sample_rate
        ),
        GenerateString(
            sample_rate=sample_rate
        ),
        GenerateString(
            sample_rate=sample_rate
        ),
        WhiteNoiseBurst(
            sample_rate=sample_rate,
            duration=duration
        ),
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
        None
    ]

    loader = SignalDataloader(signal_classes,
        priors,
        waveforms,
        extra_kwargs,
        data_dir=args.data_dir,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        psd_length=psd_length,
        fduration=fduration,
        fftlength=fftlength,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        num_workers=num_workers,
        data_saving_file=data_saving_file
    )
    test_loader = loader.test_dataloader()

    for batch in test_loader:
        [batch] = batch
        waveforms, params, ras, decs, phics = loader.generate_waveforms(batch.shape[0])
        batch = batch.to(device)
        x,snr = loader.multiInject_SNR(waveforms, batch)
        labels = torch.cat([(i+1)*torch.ones(loader.num_per_class[i]) for i in range(loader.num_classes)])
        break

    # Load feature extractor
    if args.embedding_model:
        embed_model = torch.jit.load(args.embedding_model)
        embed_model.eval()
        embed_model.to(device=device)
    else:
        ckpt = "output/S4_SimCLR_multiSignalAndBkg/lightning_logs/8wuhxd59/checkpoints/47-2400.ckpt"
        cfg_path = "output/S4_SimCLR_multiSignalAndBkg/config.yaml"
        with open(cfg_path, "r") as fin:
            cfg = yaml.load(fin, yaml.FullLoader)
        embed_model = Crayon.load_from_checkpoint(ckpt, **cfg['model']['init_args']).model
        embed_model.eval()
        embed_model.to(device=device)

    metric_model = torch.jit.load(args.fm_model)
    metric_model.eval()
    metric_model.to(device=device)


    def add_freq_corr(x, embedding):
        # batch: (batch_size, 2, time_series_length)

        # FFT along the last axis
        H = np.fft.rfft(x[:, 0, :], axis=-1)  # Hanford
        L = np.fft.rfft(x[:, 1, :], axis=-1)  # Livingston

        # Complex dot product over frequency bins
        numerator = np.sum(H * np.conj(L), axis=-1)

        # Compute norms
        norm_H = np.linalg.norm(H, axis=-1)
        norm_L = np.linalg.norm(L, axis=-1)

        # Normalize and take real part
        rho_complex = numerator / (norm_H * norm_L + 1e-8)  # avoid division by zero
        rho_real = np.real(rho_complex)[..., np.newaxis]    # shape: (batch_size, 1)

        return np.concatenate((embedding, rho_real),axis=1)


    # Containers to store anomaly scores and binary labels
    all_scores = []
    all_binary_labels = []
    all_labels = []
    all_snrs = []
    all_embeddings = []

    # Iterate over the test data loader
    niter = 1
    for ib in range(niter):
        print(f"iter {ib+1}/{niter}")
        for batch in tqdm(test_loader):
            [batch] = batch
            # Generate the corresponding waveforms
            waveforms_, params, ras, decs, phics = loader.generate_waveforms(batch.shape[0])
            batch = batch.to(device)
            waveforms=[]
            for w in waveforms:
                waveforms.append(w.to(device))

            # Process the waveforms into the required input format
            x, snrs = loader.multiInject_SNR(waveforms, batch)
            snrs = snrs/sample_rate # TODO: why is this necessary lmao

            # Reconstruct the ground truth multi-class labels
            labels = torch.cat([(i+1)*torch.ones(loader.num_per_class[i]) for i in range(loader.num_classes)]).to(device)

            with torch.no_grad():
                z = embed_model(x.to(device))
                if args.use_freq_correlation: z = add_freq_corr(x, z)
                scores = metric_model(z).squeeze().cpu().numpy()
                # The combined model directly outputs the anomaly score
                #scores = combined_model(x.to(device)).squeeze()  # (batch_size,)
                #scores = scores.cpu().numpy()
                # Invert the scores so that higher values correspond to more anomalous behavior
                scores = -scores

            # Convert the multi-class labels to binary labels:
            # Treat "Background" (assumed to be label 8) as 0 (normal) and all other classes as 1 (anomaly)
            binary_labels = (labels != 8).cpu().numpy()

            all_scores.append(scores)
            all_binary_labels.append(binary_labels)
            all_labels.append(labels.cpu().numpy())
            all_snrs.append(snrs.cpu().numpy())
            all_embeddings.append(z.cpu().numpy())

    # Concatenate scores and labels across batches
    all_scores = np.concatenate(all_scores)
    all_binary_labels = np.concatenate(all_binary_labels)
    all_labels = np.concatenate(all_labels)
    all_snrs = np.concatenate(all_snrs)
    all_embeddings = np.concatenate(all_embeddings)

    np.save(f'{args.output}/nf_scores.npy',all_scores)
    np.save(f'{args.output}/labels.npy',all_labels)
    np.save(f'{args.output}/snrs.npy',all_snrs)
    np.save(f'{args.output}/embeddings.npy',all_embeddings)

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
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colors for each anomaly class

    for i, anomaly_class in enumerate(signal_classes[:-1]):  # Exclude "Background"
        # The anomaly class numeric label is i+1.
        anomaly_val = i + 1
        # Filter to only examples that are either the current anomaly class or Background (label 8).
        idx = np.where((all_labels == anomaly_val) | (all_labels == 8))[0]
        if idx.size == 0:
            continue  # skip if no examples
        scores_i = all_scores[idx]
        # Create binary ground truth: positive if the example is of the current anomaly class, 0 if background.
        binary_labels = (all_labels[idx] == anomaly_val).astype(int)

        # Compute the ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(binary_labels, scores_i)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
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

    segment_duration = 0.5
    year_sec = 3.154e7

    fraction_1yr = (1.0 / year_sec) * segment_duration
    # Extract background-only data
    bg_mask = (all_labels == 8)
    bg_scores = all_scores[bg_mask]
    threshold_1yr = np.quantile(bg_scores,1-fraction_1yr)

    ###################
    ## Make corner plot
    N = all_embeddings.shape[1]
    labs_uniq = sorted(list(set(all_labels)))
    fig,axes = plt.subplots(N,N,figsize=(20,20))

    for i in range(all_embeddings.shape[1]):
        for j in range(i+1,all_embeddings.shape[1]):
            plt.sca(axes[i,j])
            plt.axis('off')

    for i in range(all_embeddings.shape[1]):
        plt.sca(axes[i,i])
        plt.xticks([])
        plt.yticks([])
        bins = 30
        for j,lab in enumerate(labs_uniq):
            h,bins,_ = plt.hist(all_embeddings[all_labels==lab][:,i],bins=bins,histtype='step',color=f"C{j}")

    for i in range(1,all_embeddings.shape[1]):
        for j in range(i):
            plt.sca(axes[i,j])
            plt.xticks([])
            plt.yticks([])
            for k,lab in enumerate(labs_uniq):
                ysel = all_embeddings[all_labels==lab]
                plt.scatter(ysel[:,j],ysel[:,i],s=2,color=f"C{k}")

    from matplotlib.patches import Patch
    plt.sca(axes[2,5])
    patches = []
    for k,lab in enumerate(labs_uniq):
        patches.append(Patch(color=f"C{k}",label=signal_classes[k]))
    plt.legend(handles=patches,ncol=2,fontsize=12)
    plt.savefig(f'{args.output}/corner_plot.png')
    plt.clf()

    ####
    # Check the performance of the NF
    plt.figure()
    for i, c in enumerate(signal_classes):
        scores_sel = all_scores[all_labels==i+1]
        plt.hist(scores_sel, bins=100, label=c, density=True, alpha=0.8, range=(0,500))

    plt.xlabel("NF log probability")
    plt.legend()
    plt.savefig(f'{args.output}/metric.png')
    plt.clf()

    # # -----------------------------------------------------------------------------
    # # 3) BIN ANOMALIES BY SNR AND COMPUTE FRACTION DETECTED (SCORE > threshold_1yr)
    # # -----------------------------------------------------------------------------
    # # We'll define 10 bins in SNR from the minimum anomaly SNR to the maximum
    # # anomaly SNR across all anomaly classes (labels 1..7).
    # anom_mask = (all_labels != 8)
    # if np.any(anom_mask):
    #     snr_min, snr_max = all_snrs[anom_mask].min(), all_snrs[anom_mask].max()
    # else:
    #     raise ValueError("No anomaly samples found in the test set!")

    # num_bins = 10
    # bin_edges = np.logspace(-2, 3, num_bins + 1)
    # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # plt.figure(figsize=(8,6))

    # for i, anom_class_name in enumerate(signal_classes[:-1]):
    #     class_label = i + 1  # classes are 1..7
    #     mask = (all_labels == class_label)
    #     if not np.any(mask):
    #         # If no samples for that class, skip
    #         continue

    #     class_scores = all_scores[mask]
    #     class_snrs = all_snrs[mask]

    #     # Bin by SNR
    #     bin_idx = np.digitize(class_snrs, bin_edges) - 1  # bin indices in [0..num_bins-1]

    #     frac_detected = []
    #     for b in range(num_bins):
    #         in_bin = (bin_idx == b)
    #         if not np.any(in_bin):
    #             frac_detected.append(np.nan)  # or 0.0 if you prefer
    #         else:
    #             # fraction that exceed threshold
    #             frac = np.mean(class_scores[in_bin] > threshold_1yr)
    #             frac_detected.append(frac)

    #     plt.plot(bin_centers, frac_detected, marker='o', label=anom_class_name)

    # plt.xlabel("SNR")
    # plt.ylabel("Fraction of events detected")
    # plt.title("Fraction of Events Detected at 1/Year FAR vs. SNR")
    # plt.ylim([0, 1.05])
    # plt.xscale('log')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'{args.output}/fraction_1overYearFAR_SNR.png')
    # plt.show()

    # bins = np.logspace(1,7,100)
    # for i,sig in enumerate(signal_classes[:-1]):
    #     h=plt.hist(all_snrs[all_labels==i+1],bins=np.logspace(-3,3,100),label=sig,histtype='step')
    # plt.xscale('log')
    # plt.xlabel("SNR")
    # plt.legend()