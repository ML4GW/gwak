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
from torch.distributions import Uniform
from ml4gw.distributions import PowerLaw
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.waveforms import SineGaussian, MultiSineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst
from scipy.signal import coherence  # not strictly needed, but harmless

from gwak.train.dataloader import SignalDataloader
from gwak.data.prior import SineGaussianBBC, MultiSineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from gwak.train.cl_models import Crayon

from gwak.train.plotting import make_corner
from gwak.train.preselection import (
    cwb_cc_rho_max_over_delay_2ifo,
    cwb_stats_and_delay_2ifo,
    _freq_corr_band,
    _coherence_simple,
)

# ======================= CONFIG BLOCK (top of script) =======================
# Define lists of thresholds; all lists must have the same length.

# Human-readable names for each configuration
THRESHOLD_NAMES = [
    "GWAK only",          # all postselection cuts None
    "GWAK + 20%",
    "GWAK + 30%",
    "GWAK + 40%",
]

# GWAK score thresholds
THRESHOLDS = [35, 18.59, 13.53, 8.38]

# FAR labels corresponding to each threshold (used in plot titles/legend)
FAR_LABELS = [
    "1/month",
    "1/month",
    "1/month",
    "1/month",
]

# Postselection thresholds; use None to disable a given cut for that config.
# For the "GWAK only" config, all of these should be None.
CC_MIN_LIST   = [None, None, None, None]
RHO_MIN_LIST  = [None, None, None, 0.3412]
SCC_MIN_LIST  = [None, 0.00023431127849997998, 0.00023682538571712, 0.00023895325063952]
EDR_MAX_LIST  = [None, 10.41687, 9.866419386482177, 9.403834180773776]

# Extra postselection variables thresholds (None => no cut applied)
COH_MEAN_MIN_LIST     = [None, 0.03112481907010074, 0.03218076229095454, 0.03309622183442108]
COH_MAX_MIN_LIST      = [None, 0.14165014922618863, 0.1506790161132812, 0.15929250121116637]
CORR_MAG_FD_MIN_LIST  = [None, 0.04881879268688244, 0.0515990199139864, 0.05405192967865194]
CORR_REAL_FD_MIN_LIST = [None, 0.04582270953727188, 0.0482541931387443, 0.050501710990046085]

# Toggle whether to do postselection at all (still compute stats if True)
DO_POSTSELECTION = True
# ===========================================================================

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


def _plot_postselection_hist(
    var_name,
    values,
    thresholds_list,
    bkg_mask,
    sig_mask,
    output_dir,
    nbins=100
):
    """
    Plot a histogram of a postselection variable with:
      - background and signal overlaid (different colours)
      - vertical lines at all non-None thresholds from thresholds_list.
    """
    if values is None:
        return

    # Only finite values
    finite = np.isfinite(values)

    bkg_vals = values[bkg_mask & finite]
    sig_vals = values[sig_mask & finite]

    if bkg_vals.size == 0 and sig_vals.size == 0:
        return

    plt.figure(figsize=(8, 6))

    # Choose common range from combined data
    all_vals = np.concatenate([bkg_vals, sig_vals]) if bkg_vals.size and sig_vals.size else \
               (bkg_vals if bkg_vals.size else sig_vals)
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    plt.hist(
        bkg_vals,
        bins=nbins,
        range=(vmin, vmax),
        alpha=0.6,
        label="Background",
        density=True,
        histtype="stepfilled"
    )
    plt.hist(
        sig_vals,
        bins=nbins,
        range=(vmin, vmax),
        alpha=0.6,
        label="Signal (all)",
        density=True,
        histtype="stepfilled"
    )

    # Unique non-None thresholds
    thr_unique = sorted({t for t in thresholds_list if t is not None})
    for thr in thr_unique:
        plt.axvline(thr, linestyle="--", color="k", alpha=0.8)
    # Add a legend entry for thresholds (if any)
    if thr_unique:
        plt.axvline(thr_unique[0], linestyle="--", color="k", alpha=0.8,
                    label="Postselection thresholds")

    plt.xlabel(var_name)
    plt.ylabel("Density")
    plt.title(f"{var_name} distribution (signal vs background)")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"hist_{var_name}.png")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.clf()


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
    # threshold now taken from THRESHOLDS list at top
    parser.add_argument('--snr-cut', type=float, default=0)
    parser.add_argument('--conditioning', type=str2bool, default=True)
    parser.add_argument('--averaging-kernel', type=int, default=1)

    # postselection options: flo/fhi still from CLI, but cuts from top-of-file config
    parser.add_argument('--flo', type=float, default=30.0)
    parser.add_argument('--fhi', type=float, default=2048.0)

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
        snr_prior=Uniform(4, 30),  # matches your current baseline
        glitch_root=f"/home/hongyin.chen/anti_gravity/gwak/gwak/output/O4b_AnalysisReady_Cat12/omicron/"
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
        # cWB stats for postselection
        'all_cc': f'{args.output}_precomputed/cc_{args.nevents}.npy',
        'all_rho': f'{args.output}_precomputed/rho_{args.nevents}.npy',
        'all_scc': f'{args.output}_precomputed/scc_{args.nevents}.npy',
        'all_edr': f'{args.output}_precomputed/edr_{args.nevents}.npy',
        # extra postselection variables
        'all_coh_mean': f'{args.output}_precomputed/coh_mean_{args.nevents}.npy',
        'all_coh_max': f'{args.output}_precomputed/coh_max_{args.nevents}.npy',
        'all_corr_mag_fd': f'{args.output}_precomputed/corr_mag_fd_{args.nevents}.npy',
        'all_corr_real_fd': f'{args.output}_precomputed/corr_real_fd_{args.nevents}.npy',
    }
    datasets = {}
    for key, filepath in filenames.items():
        if os.path.exists(filepath):
            print(f"Loading {filepath}...")
            datasets[key] = np.load(filepath)

    if datasets and all(k in datasets for k in ['all_binary_labels','all_scores','all_embeddings','all_labels','all_snrs']):
        all_binary_labels = datasets['all_binary_labels']
        all_scores = datasets['all_scores']
        all_embeddings = datasets['all_embeddings']
        all_labels = datasets['all_labels']
        all_snrs = datasets['all_snrs']
        all_context = datasets.get('all_context', np.array([]))

        # Load cWB stats & extra variables if present (needed for postselection)
        all_cc  = datasets.get('all_cc',  None)
        all_rho = datasets.get('all_rho', None)
        all_scc = datasets.get('all_scc', None)
        all_edr = datasets.get('all_edr', None)

        all_coh_mean     = datasets.get('all_coh_mean',    None)
        all_coh_max      = datasets.get('all_coh_max',     None)
        all_corr_mag_fd  = datasets.get('all_corr_mag_fd', None)
        all_corr_real_fd = datasets.get('all_corr_real_fd', None)

        if DO_POSTSELECTION:
            # If postselection requested but any needed arrays missing, force recomputation
            needed = [all_cc, all_rho, all_scc, all_edr,
                      all_coh_mean, all_coh_max, all_corr_mag_fd, all_corr_real_fd]
            if any(x is None for x in needed):
                raise ValueError("Postselection requested but some postselection arrays are not cached. "
                                 "Delete precomputed files and rerun without cache to compute them.")
    else:
        all_binary_labels = []
        all_scores = []
        all_embeddings = []
        all_labels = []
        all_context = []
        all_snrs = []
        all_cc_list  = []
        all_rho_list = []
        all_scc_list = []
        all_edr_list = []
        all_coh_mean_list     = []
        all_coh_max_list      = []
        all_corr_mag_fd_list  = []
        all_corr_real_fd_list = []

        n_iter = int(args.nevents/batch_size)
        test_loader = loader.test_dataloader()
        test_iter = iter(test_loader)
        for i in tqdm(range(n_iter), desc="Processing batches"):
            clean_batch, glitch_batch = next(test_iter)
            clean_batch = clean_batch.to(device)
            glitch_batch = glitch_batch.to(device)

            processed, labels, snrs = loader.on_after_batch_transfer(
                [clean_batch, glitch_batch],
                None,
                local_test=True)

            embeddings = embed_model(processed)

            if args.conditioning:
                context = frequency_cos_similarity(processed)
                all_context.append(context.detach().cpu().numpy())
                scores = metric_model(embeddings, context=context).detach().cpu().numpy() * (-1)
            else:
                scores = metric_model(embeddings).detach().cpu().numpy() * (-1)

            # add averaging kernel
            if args.averaging_kernel > 1:
                kernel = np.ones((args.averaging_kernel,)) / args.averaging_kernel
                scores = np.convolve(scores.flatten(), kernel, mode='valid').reshape(-1, scores.shape[1])

            embeddings_np = embeddings.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            snrs_np = snrs.detach().cpu().numpy()

            binary_labels = (~np.isin(labels_np, background_labels)).astype(int)

            all_binary_labels.append(binary_labels)
            all_labels.append(labels_np)
            all_scores.append(scores)
            all_embeddings.append(embeddings_np)
            all_snrs.append(snrs_np)

            # compute cWB stats + extra postselection variables if requested
            if DO_POSTSELECTION:
                processed_np = processed.detach().cpu().numpy()  # (B, 2, T)
                B = processed_np.shape[0]
                cc_batch    = np.zeros(B, dtype=np.float32)
                rho_batch   = np.zeros(B, dtype=np.float32)
                scc_batch   = np.zeros(B, dtype=np.float32)
                edr_batch   = np.zeros(B, dtype=np.float32)
                coh_mean_b  = np.zeros(B, dtype=np.float32)
                coh_max_b   = np.zeros(B, dtype=np.float32)
                corr_mag_b  = np.zeros(B, dtype=np.float32)
                corr_real_b = np.zeros(B, dtype=np.float32)

                for j in range(B):
                    h = processed_np[j, 0, :]
                    l = processed_np[j, 1, :]

                    # --- 1) cc, rho, scc, edr from delay scan (max over cc)
                    cc, rho, scc_v, edr = cwb_cc_rho_max_over_delay_2ifo(
                        h, l,
                        sample_rate,         # fs
                        args.flo, args.fhi,  # band
                        max_tau=0.010,
                        n_tau=81,
                        nperseg=256,
                        noverlap=128,
                        nfft=4096,
                        window_time=True,
                        use_prewhitened=True,
                    )
                    cc_batch[j]  = cc
                    rho_batch[j] = rho
                    scc_batch[j] = scc_v
                    edr_batch[j] = edr

                    # --- 2) tau_star from cwb_stats_and_delay_2ifo
                    _cc2, _rho2, _scc2, _edr2, tau_star = cwb_stats_and_delay_2ifo(
                        h, l,
                        sample_rate,         # fs
                        args.flo, args.fhi,  # band
                        max_tau=0.010,
                        n_tau=81,
                        nperseg=256,
                        noverlap=128,
                        nfft=4096,
                        window_time=True,
                        use_prewhitened=True,
                    )

                    # --- 3) coherence in band
                    coh_mean, coh_max = _coherence_simple(
                        h, l,
                        sample_rate,
                        args.flo, args.fhi,
                        nperseg=256,
                        noverlap=128,
                    )
                    coh_mean_b[j] = coh_mean
                    coh_max_b[j]  = coh_max

                    # --- 4) frequency-domain corr using tau_star
                    corr_mag_fd, corr_real_fd, corr_imag_fd, corr_phase_fd = _freq_corr_band(
                        h, l,
                        sample_rate,
                        args.flo, args.fhi,
                        use_prewhitened=True,
                        nperseg=256,
                        noverlap=128,
                        nfft=4096,
                        window_time=True,
                        tau=tau_star,
                    )
                    corr_mag_b[j]  = corr_mag_fd
                    corr_real_b[j] = corr_real_fd

                all_cc_list.append(cc_batch)
                all_rho_list.append(rho_batch)
                all_scc_list.append(scc_batch)
                all_edr_list.append(edr_batch)
                all_coh_mean_list.append(coh_mean_b)
                all_coh_max_list.append(coh_max_b)
                all_corr_mag_fd_list.append(corr_mag_b)
                all_corr_real_fd_list.append(corr_real_b)

            del clean_batch, glitch_batch, processed, embeddings, binary_labels, labels, scores, snrs
            torch.cuda.empty_cache()

        if args.conditioning:
            all_context = np.concatenate(all_context, axis=0) if len(all_context) > 0 else np.array([])
        all_binary_labels = np.concatenate(all_binary_labels, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_snrs = np.concatenate(all_snrs, axis=0)

        if DO_POSTSELECTION and len(all_cc_list) > 0:
            all_cc  = np.concatenate(all_cc_list,  axis=0)
            all_rho = np.concatenate(all_rho_list, axis=0)
            all_scc = np.concatenate(all_scc_list, axis=0)
            all_edr = np.concatenate(all_edr_list, axis=0)
            all_coh_mean     = np.concatenate(all_coh_mean_list,    axis=0)
            all_coh_max      = np.concatenate(all_coh_max_list,     axis=0)
            all_corr_mag_fd  = np.concatenate(all_corr_mag_fd_list, axis=0)
            all_corr_real_fd = np.concatenate(all_corr_real_fd_list, axis=0)
        else:
            all_cc = all_rho = all_scc = all_edr = None
            all_coh_mean = all_coh_max = all_corr_mag_fd = all_corr_real_fd = None

        os.makedirs(f"{args.output}_precomputed", exist_ok=True)
        np.save(f'{args.output}_precomputed/context_{args.nevents}.npy', all_context)
        np.save(f'{args.output}_precomputed/binary_labels_{args.nevents}.npy', all_binary_labels)
        np.save(f'{args.output}_precomputed/labels_{args.nevents}.npy', all_labels)
        np.save(f'{args.output}_precomputed/scores_{args.nevents}.npy', all_scores)
        np.save(f'{args.output}_precomputed/embeddings_{args.nevents}.npy', all_embeddings)
        np.save(f'{args.output}_precomputed/snrs_{args.nevents}.npy', all_snrs)

        if DO_POSTSELECTION and all_cc is not None:
            np.save(f'{args.output}_precomputed/cc_{args.nevents}.npy',  all_cc)
            np.save(f'{args.output}_precomputed/rho_{args.nevents}.npy', all_rho)
            np.save(f'{args.output}_precomputed/scc_{args.nevents}.npy', all_scc)
            np.save(f'{args.output}_precomputed/edr_{args.nevents}.npy', all_edr)
            np.save(f'{args.output}_precomputed/coh_mean_{args.nevents}.npy',    all_coh_mean)
            np.save(f'{args.output}_precomputed/coh_max_{args.nevents}.npy',     all_coh_max)
            np.save(f'{args.output}_precomputed/corr_mag_fd_{args.nevents}.npy', all_corr_mag_fd)
            np.save(f'{args.output}_precomputed/corr_real_fd_{args.nevents}.npy', all_corr_real_fd)

    # PLOT SNR HISTS
    # Find unique labels, excluding 10, 11, 12
    unique_labels = [label for label in np.unique(all_labels) if label not in [10, 11, 12]]

    # Create a figure
    plt.figure(figsize=(12, 7))

    # Plot a histogram for each allowed label
    for label in unique_labels:
        snrs = all_snrs[all_labels == label]
        name = signal_classes[int(label) - 1]
        snr_min = np.min(snrs)
        snr_max = np.max(snrs)
        snr_mean = np.mean(snrs)
        plt.hist(snrs, bins=50, alpha=0.5, label=f'{name} (min={snr_min:.1f}, max={snr_max:.1f}, mean={snr_mean:.1f})')

    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.title('SNR Distribution by Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{args.output}/snrs-label.png')

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

    # ------------------------------------------------------------------
    # Histograms for postselection variables (signal vs background)
    # with vertical lines at thresholds from the config lists
    # ------------------------------------------------------------------
    postsel_outdir = args.output  # use same output dir for now

    if DO_POSTSELECTION and all_cc is not None:
        _plot_postselection_hist(
            var_name="cc",
            values=all_cc,
            thresholds_list=CC_MIN_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )
        _plot_postselection_hist(
            var_name="rho",
            values=all_rho,
            thresholds_list=RHO_MIN_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )
        _plot_postselection_hist(
            var_name="scc",
            values=all_scc,
            thresholds_list=SCC_MIN_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )
        _plot_postselection_hist(
            var_name="edr",
            values=all_edr,
            thresholds_list=EDR_MAX_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )
        _plot_postselection_hist(
            var_name="coh_mean",
            values=all_coh_mean,
            thresholds_list=COH_MEAN_MIN_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )
        _plot_postselection_hist(
            var_name="coh_max",
            values=all_coh_max,
            thresholds_list=COH_MAX_MIN_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )
        _plot_postselection_hist(
            var_name="corr_mag_fd",
            values=all_corr_mag_fd,
            thresholds_list=CORR_MAG_FD_MIN_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )
        _plot_postselection_hist(
            var_name="corr_real_fd",
            values=all_corr_real_fd,
            thresholds_list=CORR_REAL_FD_MIN_LIST,
            bkg_mask=bkg_mask,
            sig_mask=sig_mask,
            output_dir=postsel_outdir,
        )

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
    plt.title(f'{args.ifos} ROC (All Signals)')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/roc_combined.png')
    plt.clf()

    # Compute and plot ROC curves for each anomaly class (all but "Background").
    plt.figure(figsize=(8,6))
    for i, anomaly_class in enumerate(signal_classes):
        # The anomaly class numeric label is i+1.
        anomaly_val = i + 1
        # Filter to only examples that are either the current anomaly class or Background.
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
    plt.title(f'{args.ifos} ROC Curves by Anomaly Class')
    plt.legend(loc='lower right')
    plt.savefig(f'{args.output}/rocs_bySignal.png')

    # ###################
    # ## Make corner plot
    fig = make_corner(all_embeddings, (all_labels-1).astype(int), return_fig=True, label_names=all_classes)
    fig.savefig(f'{args.output}/corner_plot.png')

    # Define custom colors
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939',
        '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#9c9ede', '#637939',
        '#e7ba52', '#ad494a'
    ]

    #####
    # Stack histograms properly
    plt.figure()

    # Collect all scores in a list
    score_list = []
    noise_score = []
    for i, c in enumerate(all_classes):
        scores_sel = all_scores[all_labels == i + 1]
        score_list.append(scores_sel)

        if c in ('Background', 'Glitch', 'FakeGlitch'):
            noise_sel = all_scores[all_labels == i + 1]
            noise_score.append(noise_sel)

    all_score = np.concatenate(score_list)
    max_noise_score = np.concatenate(noise_score).max()
    range_min, range_max = all_score.min(), all_score.max()

    # Plot all at once
    plt.hist(
        score_list,
        bins=100,
        label=all_classes,
        alpha=0.8,
        range=(range_min, range_max),
        stacked=True,
        color=custom_colors[:len(all_classes)]  # trim color list if needed
    )
    plt.axvline(x=max_noise_score, color='r', linestyle='--', label=f'Max noise score {max_noise_score:.2f}')
    plt.xlabel(f"{args.ifos} NF log probability")
    plt.legend()
    plt.savefig(f'{args.output}/metric.png')
    plt.clf()

    # -----------------------------------------------------------------------------
    # GWAK + (possibly trivial) POSTSELECTION efficiencies vs SNR
    # One plot per configuration (threshold + postselection cuts)
    # -----------------------------------------------------------------------------

    anom_mask = ~np.isin(all_labels, background_labels)
    if np.any(anom_mask):
        snr_min, snr_max = all_snrs[anom_mask].min(), all_snrs[anom_mask].max()
    else:
        raise ValueError("No anomaly samples found in the test set!")

    num_bins = 10
    bin_edges = np.linspace(4, 30, num_bins + 1)  # 10 bins between 4 and 30
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_cfg = len(THRESHOLDS)
    assert len(THRESHOLD_NAMES) == n_cfg
    assert len(FAR_LABELS) == n_cfg
    assert len(CC_MIN_LIST) == n_cfg == len(RHO_MIN_LIST) == len(SCC_MIN_LIST) == len(EDR_MAX_LIST)
    assert len(COH_MEAN_MIN_LIST) == n_cfg == len(COH_MAX_MIN_LIST) == len(CORR_MAG_FD_MIN_LIST) == len(CORR_REAL_FD_MIN_LIST)

    # -------------------------------------------------------------
    # One plot per configuration (GWAK (+ postselection))
    # -------------------------------------------------------------
    for cfg_idx in range(n_cfg):
        threshold = THRESHOLDS[cfg_idx]
        name_cfg = THRESHOLD_NAMES[cfg_idx]
        far_label = FAR_LABELS[cfg_idx]

        cc_min_cfg          = CC_MIN_LIST[cfg_idx]
        rho_min_cfg         = RHO_MIN_LIST[cfg_idx]
        scc_min_cfg         = SCC_MIN_LIST[cfg_idx]
        edr_max_cfg         = EDR_MAX_LIST[cfg_idx]
        coh_mean_min_cfg    = COH_MEAN_MIN_LIST[cfg_idx]
        coh_max_min_cfg     = COH_MAX_MIN_LIST[cfg_idx]
        corr_mag_fd_min_cfg = CORR_MAG_FD_MIN_LIST[cfg_idx]
        corr_real_fd_min_cfg = CORR_REAL_FD_MIN_LIST[cfg_idx]

        # Build config-specific postselection mask (None => no cut)
        if DO_POSTSELECTION:
            post_mask_cfg = np.ones_like(all_labels, dtype=bool)

            # Original cWB stats
            if cc_min_cfg is not None:
                post_mask_cfg &= (all_cc  >= cc_min_cfg)
            if rho_min_cfg is not None:
                post_mask_cfg &= (all_rho >= rho_min_cfg)
            if scc_min_cfg is not None:
                post_mask_cfg &= (all_scc >= scc_min_cfg)
            if edr_max_cfg is not None:
                post_mask_cfg &= (all_edr <= edr_max_cfg)

            # Extra variables; None => no cut
            if coh_mean_min_cfg is not None:
                post_mask_cfg &= (all_coh_mean >= coh_mean_min_cfg)
            if coh_max_min_cfg is not None:
                post_mask_cfg &= (all_coh_max >= coh_max_min_cfg)
            if corr_mag_fd_min_cfg is not None:
                post_mask_cfg &= (all_corr_mag_fd >= corr_mag_fd_min_cfg)
            if corr_real_fd_min_cfg is not None:
                post_mask_cfg &= (all_corr_real_fd >= corr_real_fd_min_cfg)
        else:
            post_mask_cfg = np.ones_like(all_labels, dtype=bool)

        plt.figure(figsize=(8,6))

        for i, anom_class_name in enumerate(signal_classes):
            class_label = i + 1  # classes are 1..7

            mask = (all_labels == class_label)
            if not np.any(mask):
                continue

            class_scores = all_scores[mask]
            class_snrs = all_snrs[mask]
            class_post_mask = post_mask_cfg[mask]  # per-event postselection mask for this config

            # Bin by SNR
            bin_idx = np.digitize(class_snrs, bin_edges) - 1

            frac_detected = []
            for b in range(num_bins):
                in_bin = (bin_idx == b)
                if not np.any(in_bin):
                    frac_detected.append(np.nan)
                else:
                    gwak_pass = class_scores[in_bin] > threshold
                    post_pass = class_post_mask[in_bin]
                    # GWAK + postselection (trivial if all cuts None)
                    frac = np.mean(gwak_pass & post_pass)
                    frac_detected.append(frac)

            plt.plot(bin_centers, frac_detected, marker='o', label=anom_class_name)

        safe_name = name_cfg.replace(" ", "_")
        plt.xlabel("SNR")
        plt.ylabel("Fraction of events detected")
        plt.title(
            f"{args.ifos} Fraction of Events Detected ({name_cfg}, "
            f"thr={threshold:.2f}, FAR={far_label}) vs. SNR"
        )
        leg = plt.legend()
        leg.set_title(f"FAR = {far_label}")
        plt.ylim([0, 1.05])
        plt.grid(True)
        plt.savefig(f'{args.output}/fraction_FAR_{safe_name}_SNR_gwak_cfg{cfg_idx}.png')
        plt.clf()