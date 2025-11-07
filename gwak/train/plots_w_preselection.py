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

from ml4gw.distributions import PowerLaw
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.waveforms import SineGaussian, MultiSineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

from gwak.train.dataloader import SignalDataloader
from gwak.data.prior import SineGaussianBBC, MultiSineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from gwak.train.cl_models import Crayon

from gwak.train.plotting import make_corner

# === NEW: preselection stats
from gwak.train.preselection import cwb_stats_2ifo, cwb_cc_rho_max_over_delay_2ifo
# ===========================

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


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
    parser.add_argument('--snr-cut', type=float, default=0)
    parser.add_argument('--conditioning', type=str2bool, default=False)
    parser.add_argument('--averaging-kernel', type=int, default=1)

    # === NEW: preselection controls & thresholds (same style as your other script)
    parser.add_argument('--flo', type=float, default=30.0)
    parser.add_argument('--fhi', type=float, default=2048.0)
    parser.add_argument('--delay-scan', action='store_true', default=True)
    parser.add_argument('--tau-max', type=float, default=0.010)
    parser.add_argument('--n-tau', type=int, default=81)
    parser.add_argument('--cc-min',  type=float, default=0.52626751)
    parser.add_argument('--rho-min', type=float, default=1.01491416)
    parser.add_argument('--scc-min', type=float, default=0.00024060)
    parser.add_argument('--edr-max', type=float, default=0.90017429)
    # =====================================================

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
        ifos=args.ifos,
        snr_prior=PowerLaw(index=3, minimum=4, maximum=30),
        glitch_root=f"/home/hongyin.chen/anti_gravity/gwak/gwak/output/O4b_AnalysisReady_Cat12/omicron/",
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

    filenames = {
        'all_context': f'{args.output}_precomputed/context_{args.nevents}.npy',
        'all_binary_labels': f'{args.output}_precomputed/binary_labels_{args.nevents}.npy',
        'all_labels': f'{args.output}_precomputed/labels_{args.nevents}.npy',
        'all_scores': f'{args.output}_precomputed/scores_{args.nevents}.npy',
        'all_embeddings': f'{args.output}_precomputed/embeddings_{args.nevents}.npy',
        'all_snrs': f'{args.output}_precomputed/snrs_{args.nevents}.npy',

        # === NEW: store stats & preselection mask too
        'all_cc':  f'{args.output}_precomputed/cc_{args.nevents}.npy',
        'all_rho': f'{args.output}_precomputed/rho_{args.nevents}.npy',
        'all_scc': f'{args.output}_precomputed/scc_{args.nevents}.npy',
        'all_edr': f'{args.output}_precomputed/edr_{args.nevents}.npy',
        'pre_mask': f'{args.output}_precomputed/preselection_mask_{args.nevents}.npy',
    }

    datasets = {}
    for key, filepath in filenames.items():
        if os.path.exists(filepath):
            print(f"Loading {filepath}...")
            datasets[key] = np.load(filepath, allow_pickle=False)

    if datasets and all(k in datasets for k in ['all_binary_labels','all_scores','all_embeddings','all_labels','all_context','all_snrs','all_cc','all_rho','all_scc','all_edr','pre_mask']):
        all_binary_labels = datasets['all_binary_labels']
        all_scores = datasets['all_scores']
        all_embeddings = datasets['all_embeddings']
        all_labels = datasets['all_labels']
        all_context = datasets['all_context']
        all_snrs = datasets['all_snrs']
        all_cc  = datasets['all_cc']
        all_rho = datasets['all_rho']
        all_scc = datasets['all_scc']
        all_edr = datasets['all_edr']
        pre_mask = datasets['pre_mask'].astype(bool)
    else:
        all_binary_labels = []
        all_scores = []
        all_embeddings = []
        all_labels = []
        all_context = []
        all_snrs = []

        # === NEW: per-sample stats & preselection mask
        all_cc, all_rho, all_scc, all_edr = [], [], [], []
        pre_mask_list = []
        # =============================================

        n_iter = int(args.nevents/batch_size)
        test_loader = loader.test_dataloader()
        test_iter = iter(test_loader)
        for i in tqdm(range(n_iter), desc="Processing batches"):
            clean_batch, glitch_batch = next(test_iter)
            clean_batch = clean_batch.to(device)
            glitch_batch = glitch_batch.to(device)

            processed, labels, snrs = loader.on_after_batch_transfer([clean_batch, glitch_batch], None,
                local_test=True)
            B = processed.shape[0]

            # === NEW: compute stats BEFORE embedder/NF
            batch_np = processed.detach().cpu().numpy().astype(np.float32)
            cc_list, rho_list, scc_list, edr_list = [], [], [], []
            for j in range(B):
                h = batch_np[j, 0, :]
                l = batch_np[j, 1, :]
                if args.delay_scan:
                    cc, rho, scc_v, edr = cwb_cc_rho_max_over_delay_2ifo(
                        h, l, sample_rate, args.flo, args.fhi,
                        max_tau=args.tau_max, n_tau=args.n_tau
                    )
                else:
                    st = cwb_stats_2ifo(h, l, sample_rate, args.flo, args.fhi)
                    cc, rho, scc_v, edr = st["cc"], st["rho"], st["scc"], st["edr"]
                cc_list.append(cc); rho_list.append(rho); scc_list.append(scc_v); edr_list.append(edr)

            cc_arr  = np.asarray(cc_list,  dtype=np.float32)
            rho_arr = np.asarray(rho_list, dtype=np.float32)
            scc_arr = np.asarray(scc_list, dtype=np.float32)
            edr_arr = np.asarray(edr_list, dtype=np.float32)

            # preselection mask (NO filtering yet for full-dataset quantities)
            batch_pre_mask = (
                (cc_arr  >= args.cc_min) &
                (rho_arr >= args.rho_min) &
                (scc_arr >= args.scc_min) &
                (edr_arr <= args.edr_max)
            )
            # =========================================

            # Now run embedder + NF as before
            embeddings = embed_model(processed)

            if args.conditioning:
                context = frequency_cos_similarity(processed)
                all_context.append(context.detach().cpu().numpy())
                scores = metric_model(embeddings, context=context).detach().cpu().numpy() * (-1)
            else:
                scores = metric_model(embeddings).detach().cpu().numpy() * (-1)

            # add averaging kernel (kept exactly as you had it)
            if args.averaging_kernel > 1:
                kernel = np.ones((args.averaging_kernel,)) / args.averaging_kernel
                scores = np.convolve(scores.flatten(), kernel, mode='valid').reshape(-1, scores.shape[1])

            embeddings = embeddings.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            binary_labels = (~np.isin(labels_np, background_labels)).astype(int)

            # accumulate (full dataset)
            all_binary_labels.append(binary_labels)
            all_labels.append(labels_np)
            all_scores.append(scores)
            all_embeddings.append(embeddings)
            all_snrs.append(snrs.detach().cpu().numpy())

            # === NEW: accumulate stats & mask in same order
            all_cc.append(cc_arr)
            all_rho.append(rho_arr)
            all_scc.append(scc_arr)
            all_edr.append(edr_arr)
            pre_mask_list.append(batch_pre_mask.astype(np.bool_))
            # ===========================================

            del clean_batch, glitch_batch, processed, embeddings, binary_labels, labels, scores, snrs
            torch.cuda.empty_cache()

        if args.conditioning:
            all_context = np.concatenate(all_context, axis=0)
        all_binary_labels = np.concatenate(all_binary_labels, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_snrs = np.concatenate(all_snrs, axis=0)

        # === NEW: finalize stats arrays & mask
        all_cc  = np.concatenate(all_cc,  axis=0)
        all_rho = np.concatenate(all_rho, axis=0)
        all_scc = np.concatenate(all_scc, axis=0)
        all_edr = np.concatenate(all_edr, axis=0)
        pre_mask = np.concatenate(pre_mask_list, axis=0).astype(bool)
        # ====================================

        os.makedirs(f"{args.output}_precomputed", exist_ok=True)
        np.save(f'{args.output}_precomputed/context_{args.nevents}.npy', all_context)
        np.save(f'{args.output}_precomputed/binary_labels_{args.nevents}.npy', all_binary_labels)
        np.save(f'{args.output}_precomputed/labels_{args.nevents}.npy', all_labels)
        np.save(f'{args.output}_precomputed/scores_{args.nevents}.npy', all_scores)
        np.save(f'{args.output}_precomputed/embeddings_{args.nevents}.npy', all_embeddings)
        np.save(f'{args.output}_precomputed/snrs_{args.nevents}.npy', all_snrs)

        # === NEW: save stats + mask
        np.save(f'{args.output}_precomputed/cc_{args.nevents}.npy',  all_cc)
        np.save(f'{args.output}_precomputed/rho_{args.nevents}.npy', all_rho)
        np.save(f'{args.output}_precomputed/scc_{args.nevents}.npy', all_scc)
        np.save(f'{args.output}_precomputed/edr_{args.nevents}.npy', all_edr)
        np.save(f'{args.output}_precomputed/preselection_mask_{args.nevents}.npy', pre_mask.astype(np.uint8))
        # =================================

    # -----------------------------
    # === NEW: Preselection efficiency vs SNR (per signal class)
    # -----------------------------
    unique_labels = [label for label in np.unique(all_labels) if label not in [10, 11, 12]]
    num_bins = 10
    bin_edges = np.linspace(4, 30, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8,6))
    for i, anom_class_name in enumerate(signal_classes):
        class_label = i + 1
        mask_sig = (all_labels == class_label)
        if not np.any(mask_sig):
            continue
        cls_snrs = all_snrs[mask_sig]
        cls_pre  = pre_mask[mask_sig]  # fraction passing preselection
        bin_idx = np.digitize(cls_snrs, bin_edges) - 1

        frac_pass = []
        for b in range(num_bins):
            in_bin = (bin_idx == b)
            if not np.any(in_bin):
                frac_pass.append(np.nan)
            else:
                frac_pass.append(np.mean(cls_pre[in_bin]))
        plt.plot(bin_centers, frac_pass, marker='o', label=anom_class_name)

    plt.xlabel("SNR")
    plt.ylabel("Preselection efficiency")
    plt.title(f"{args.ifos} Preselection efficiency vs SNR")
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.output}/efficiency_preselection_SNR.png')
    plt.clf()

    # ------------- (existing) PLOT SNR HISTS -------------
    plt.figure(figsize=(12, 7))
    for label in unique_labels:
        snrs = all_snrs[all_labels == label]
        name = all_classes[int(label) - 1]
        snr_min = np.min(snrs)
        snr_max = np.max(snrs)
        snr_mean = np.mean(snrs)
        plt.hist(snrs, bins=50, alpha=0.5, label=f'{name} (min={snr_min:.1f}, max={snr_max:.1f}, mean={snr_mean:.1f})')
    plt.xlabel('SNR'); plt.ylabel('Count'); plt.title('SNR Distribution by Class')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f'{args.output}/snrs-label.png'); plt.clf()

    ########### PLOT CONTEXT
    bkg_mask = all_binary_labels == 0
    sig_mask = all_binary_labels == 1

    if 'all_context' in locals() and all_context.size:
        plt.figure()
        plt.hist(all_context[bkg_mask], bins=100, alpha=0.6, label="background", density=True)
        plt.hist(all_context[sig_mask], bins=100, alpha=0.6, label="signal", density=True)
        plt.xlabel("context"); plt.ylabel("Density"); plt.legend()
        plt.title("Histogram of Context Values")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f'{args.output}/context.png'); plt.clf()

    # ===============================
    # ROC and AUC on FULL dataset (existing)
    # ===============================
    fpr, tpr, thresholds = roc_curve(all_binary_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{args.ifos} ROC (All Signals)')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/roc_combined.png'); plt.clf()

    # Per-class ROCs (existing)
    plt.figure(figsize=(8,6))
    for i, anomaly_class in enumerate(signal_classes):
        anomaly_val = i + 1
        idx = np.where(np.isin(all_labels, background_labels + [anomaly_val]))[0]
        if idx.size == 0:
            continue
        scores_i = all_scores[idx]
        binary_labels_i = (all_labels[idx] == anomaly_val).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels_i, scores_i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{anomaly_class} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{args.ifos} ROC Curves by Anomaly Class')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/rocs_bySignal.png'); plt.clf()

    # ###################
    # ## Make corner plot (existing)
    fig = make_corner(all_embeddings, (all_labels-1).astype(int), return_fig=True, label_names=all_classes)
    fig.savefig(f'{args.output}/corner_plot.png')

    # Define custom colors (existing)
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939',
        '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#9c9ede', '#637939',
        '#e7ba52', '#ad494a'
    ]

    # Metric histogram (existing)
    plt.figure()
    score_list = []
    for i, c in enumerate(all_classes):
        scores_sel = all_scores[all_labels == i + 1]
        score_list.append(scores_sel)
    plt.hist(score_list, bins=100, label=all_classes, alpha=0.8, range=(0, 500),
             stacked=True, color=custom_colors[:len(all_classes)])
    plt.xlabel(f"{args.ifos} NF log probability")
    plt.legend()
    plt.savefig(f'{args.output}/metric.png')
    plt.clf()

    # -----------------------------------------------------------------------------
    # 3) BIN ANOMALIES BY SNR AND COMPUTE FRACTION DETECTED (SCORE > threshold_1yr)
    # -----------------------------------------------------------------------------
    threshold_1yr = args.threshold_1yr

    anom_mask = ~np.isin(all_labels, background_labels)
    if np.any(anom_mask):
        snr_min, snr_max = all_snrs[anom_mask].min(), all_snrs[anom_mask].max()
    else:
        raise ValueError("No anomaly samples found in the test set!")

    num_bins = 10
    bin_edges = np.linspace(4, 30, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # === (existing) fraction vs SNR on FULL dataset
    plt.figure(figsize=(8,6))
    for i, anom_class_name in enumerate(signal_classes):
        class_label = i + 1
        mask = (all_labels == class_label)
        if not np.any(mask):
            continue
        class_scores = all_scores[mask]
        class_snrs = all_snrs[mask]

        bin_idx = np.digitize(class_snrs, bin_edges) - 1
        frac_detected = []
        for b in range(num_bins):
            in_bin = (bin_idx == b)
            if not np.any(in_bin):
                frac_detected.append(np.nan)
            else:
                frac_detected.append(np.mean(class_scores[in_bin] > threshold_1yr))
        plt.plot(bin_centers, frac_detected, marker='o', label=anom_class_name)
    plt.xlabel("SNR")
    plt.ylabel("Fraction of events detected")
    plt.title(f"{args.ifos} Fraction of Events Detected at 1/Year FAR vs. SNR (FULL)")
    plt.ylim([0, 1.05])
    plt.legend(); plt.grid(True)
    plt.savefig(f'{args.output}/fraction_1overYearFAR_SNR.png'); plt.clf()

    # === NEW: fraction vs SNR CONDITIONED on passing preselection
    plt.figure(figsize=(8,6))
    for i, anom_class_name in enumerate(signal_classes):
        class_label = i + 1
        mask = (all_labels == class_label)
        if not np.any(mask):
            continue
        cls_scores = all_scores[mask]
        cls_snrs   = all_snrs[mask]
        cls_pre    = pre_mask[mask]

        bin_idx = np.digitize(cls_snrs, bin_edges) - 1
        frac_detected_pre = []
        for b in range(num_bins):
            in_bin = (bin_idx == b)
            if not np.any(in_bin):
                frac_detected_pre.append(np.nan)
            else:
                # efficiency wrt PRESELECTED set (conditional)
                denom = np.sum(cls_pre[in_bin])
                if denom == 0:
                    frac_detected_pre.append(np.nan)
                else:
                    num = np.sum((cls_scores[in_bin] > threshold_1yr) & cls_pre[in_bin])
                    frac_detected_pre.append(num / denom)
        plt.plot(bin_centers, frac_detected_pre, marker='o', label=anom_class_name)

    plt.xlabel("SNR")
    plt.ylabel("Fraction detected (conditional on preselection)")
    plt.title(f"{args.ifos} NF Detection vs SNR (WRT PRESELECTED)")
    plt.ylim([0, 1.05])
    plt.legend(); plt.grid(True)
    plt.savefig(f'{args.output}/fraction_1overYearFAR_SNR_preselected.png'); plt.clf()
# ============================================================
# NEW: Duplicate all downstream plots for PRESELECTED-ONLY set
# ============================================================

# Build preselected-only views
mask_pre = pre_mask
if mask_pre.sum() == 0:
    print("WARNING: No samples pass preselection; skipping preselected-only plots.")
else:
    all_scores_pre         = all_scores[mask_pre]
    all_labels_pre         = all_labels[mask_pre]
    all_binary_labels_pre  = all_binary_labels[mask_pre]
    all_snrs_pre           = all_snrs[mask_pre]
    if 'all_context' in locals() and isinstance(all_context, np.ndarray) and all_context.size:
        all_context_pre = all_context[mask_pre]
    else:
        all_context_pre = None

    # --- SNR histograms (preselected only) ---
    unique_labels_pre = [label for label in np.unique(all_labels_pre) if label not in [10, 11, 12]]
    plt.figure(figsize=(12, 7))
    for label in unique_labels_pre:
        snrs = all_snrs_pre[all_labels_pre == label]
        name = all_classes[int(label) - 1]
        snr_min = np.min(snrs); snr_max = np.max(snrs); snr_mean = np.mean(snrs)
        plt.hist(snrs, bins=50, alpha=0.5,
                 label=f'{name} (min={snr_min:.1f}, max={snr_max:.1f}, mean={snr_mean:.1f})')
    plt.xlabel('SNR'); plt.ylabel('Count')
    plt.title('SNR Distribution by Class (Preselected)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f'{args.output}/snrs-label_preselected.png'); plt.clf()

    # --- Context histogram (preselected only) ---
    if all_context_pre is not None:
        bkg_mask_pre = all_binary_labels_pre == 0
        sig_mask_pre = all_binary_labels_pre == 1
        plt.figure()
        plt.hist(all_context_pre[bkg_mask_pre], bins=100, alpha=0.6, label="background", density=True)
        plt.hist(all_context_pre[sig_mask_pre], bins=100, alpha=0.6, label="signal", density=True)
        plt.xlabel("context"); plt.ylabel("Density"); plt.legend()
        plt.title("Histogram of Context Values (Preselected)")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f'{args.output}/context_preselected.png'); plt.clf()

    # --- Combined ROC (preselected only) ---
    fpr_p, tpr_p, _ = roc_curve(all_binary_labels_pre, all_scores_pre)
    roc_auc_p = auc(fpr_p, tpr_p)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_p, tpr_p, lw=2, label=f'ROC (AUC={roc_auc_p:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{args.ifos} ROC (All Signals, Preselected)')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/roc_combined_preselected.png'); plt.clf()

    # --- Per-class ROCs (preselected only) ---
    plt.figure(figsize=(8,6))
    for i, anomaly_class in enumerate(signal_classes):
        anomaly_val = i + 1
        idx = np.where(np.isin(all_labels_pre, background_labels + [anomaly_val]))[0]
        if idx.size == 0:
            continue
        scores_i = all_scores_pre[idx]
        binary_labels_i = (all_labels_pre[idx] == anomaly_val).astype(int)
        fpr_i, tpr_i, _ = roc_curve(binary_labels_i, scores_i)
        roc_auc_i = auc(fpr_i, tpr_i)
        plt.plot(fpr_i, tpr_i, lw=2, label=f'{anomaly_class} (AUC = {roc_auc_i:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{args.ifos} ROC Curves by Anomaly Class (Preselected)')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/rocs_bySignal_preselected.png'); plt.clf()

    # --- Corner plot (preselected only) ---
    fig_pre = make_corner(all_embeddings[mask_pre], (all_labels[mask_pre]-1).astype(int),
                          return_fig=True, label_names=all_classes)
    fig_pre.savefig(f'{args.output}/corner_plot_preselected.png')

    # --- Metric / NF log-prob histogram (preselected only) ---
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939',
        '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#9c9ede', '#637939',
        '#e7ba52', '#ad494a'
    ]

    plt.figure()
    score_list_pre = []
    labels_with_max_pre = []

    for i, cls_name in enumerate(all_classes):
        # gather preselected scores for class i (labels are 1-based)
        scores_sel_pre = all_scores_pre[all_labels_pre == (i + 1)]
        scores_vec = np.ravel(scores_sel_pre)  # ensure 1D

        if scores_vec.size == 0:
            # keep empty array so bins align across classes; label shows n=0
            score_list_pre.append(scores_vec)
            labels_with_max_pre.append(f"{cls_name} (n=0)")
            continue

        score_list_pre.append(scores_vec)
        labels_with_max_pre.append(f"{cls_name} (max={scores_vec.max():.1f}, n={scores_vec.size})")

    plt.hist(
        score_list_pre,
        bins=100,
        label=labels_with_max_pre,
        alpha=0.8,
        range=(0, 500),
        stacked=True,
        color=custom_colors[:len(score_list_pre)]
    )
    plt.xlabel(f"{args.ifos} NF log probability")
    plt.legend(title="Class (max, n)")
    plt.title("NF log probability (Preselected)")
    plt.tight_layout()
    plt.savefig(f'{args.output}/metric_preselected.png')
    plt.clf()