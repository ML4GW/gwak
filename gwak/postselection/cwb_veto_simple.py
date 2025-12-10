#!/usr/bin/env python3
"""
Simple cWB Veto Framework - Clean Implementation

Combines:
1. Whitening scheme 
2. Data loading from read_strain_data.py
3. cWB features from extract_cwb_features_with_gaussianity_crosscorr_null_corr.py

Evaluates 8 key metrics: rho, edr, cc, scc, coh_max, coh_mean, corr_mag_fd, corr_real_fd
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.fft import rfft, rfftfreq
from scipy.signal import welch, coherence
from tqdm import tqdm

from ml4gw.transforms import SpectralDensity, Whiten

# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 4096
HIGHPASS = 30
FFTLENGTH = 2
FDURATION = 2

# cWB metric computation parameters
BAND = (32.0, 2048.0)  # Frequency band [Hz]
NPERSEG = 256
NOVERLAP = 128
NFFT = 4096
USE_DELAY_SCAN = True
TAU_MAX = 0.010
N_TAU = 81
EPS = 1e-12

# Data loading parameters (from read_strain_data.py)
PSD_LENGTH = 64
FFT_LENGTH = 1
KERNEL = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Whitening (from whitening.ipynb)
# ============================================================================

def whiten_data(h1_strain, l1_strain, sample_rate=SAMPLE_RATE,
                fduration=FDURATION, fftlength=FFTLENGTH, highpass=HIGHPASS):
    """
    Whiten strain data using ml4gw pipeline from whitening.ipynb.

    Processes each channel independently to avoid tensor dimension issues.
    """
    #print(f"\n[DEBUG whiten_data] Input shapes: h1_strain={h1_strain.shape}, l1_strain={l1_strain.shape}")

    # Whiten H1 separately
    h1_tensor = torch.from_numpy(h1_strain).float().to(DEVICE)
    psd_samples = int(PSD_LENGTH * sample_rate)
    #print(f"[DEBUG whiten_data] psd_samples={psd_samples}, h1_tensor.shape={h1_tensor.shape}")

    # H1 whitening
    psd_data_h1 = h1_tensor[:psd_samples]
    signal_data_h1 = h1_tensor[psd_samples:]
    #print(f"[DEBUG whiten_data] psd_data_h1.shape={psd_data_h1.shape}, signal_data_h1.shape={signal_data_h1.shape}")

    whitener = Whiten(fduration=fduration, sample_rate=sample_rate, highpass=highpass).to(DEVICE)
    spectral_density = SpectralDensity(sample_rate=sample_rate, fftlength=fftlength,
                                       overlap=None, average="median").to(DEVICE)

    with torch.no_grad():
        psd_input_h1 = psd_data_h1.unsqueeze(0).double()
        #print(f"[DEBUG whiten_data] psd_input_h1.shape (before spectral_density)={psd_input_h1.shape}")

        psd_h1 = spectral_density(psd_input_h1)
        #print(f"[DEBUG whiten_data] psd_h1.shape (after spectral_density)={psd_h1.shape}")

        psd_h1_squeezed = psd_h1.squeeze(0)
        #print(f"[DEBUG whiten_data] psd_h1_squeezed.shape={psd_h1_squeezed.shape}")

        signal_input_h1 = signal_data_h1.unsqueeze(0).double()
        #print(f"[DEBUG whiten_data] signal_input_h1.shape (before whitener)={signal_input_h1.shape}")

        psd_h1_input = psd_h1_squeezed.unsqueeze(0)
        #print(f"[DEBUG whiten_data] psd_h1_input.shape (for whitener)={psd_h1_input.shape}")

        h1_whitened_tensor = whitener(signal_input_h1, psd_h1_input)
        #print(f"[DEBUG whiten_data] h1_whitened_tensor.shape (after whitener)={h1_whitened_tensor.shape}")

        # Use squeeze() without arguments to remove ALL size-1 dimensions
        h1_whitened = h1_whitened_tensor.squeeze().cpu().numpy()
        #print(f"[DEBUG whiten_data] h1_whitened.shape (final numpy)={h1_whitened.shape}")

    # L1 whitening
    l1_tensor = torch.from_numpy(l1_strain).float().to(DEVICE)
    psd_data_l1 = l1_tensor[:psd_samples]
    signal_data_l1 = l1_tensor[psd_samples:]
    #print(f"[DEBUG whiten_data] psd_data_l1.shape={psd_data_l1.shape}, signal_data_l1.shape={signal_data_l1.shape}")

    with torch.no_grad():
        psd_l1 = spectral_density(psd_data_l1.unsqueeze(0).double()).squeeze(0)
        #print(f"[DEBUG whiten_data] psd_l1.shape={psd_l1.shape}")

        l1_whitened_tensor = whitener(signal_data_l1.unsqueeze(0).double(), psd_l1.unsqueeze(0))
        #print(f"[DEBUG whiten_data] l1_whitened_tensor.shape={l1_whitened_tensor.shape}")

        # Use squeeze() without arguments to remove ALL size-1 dimensions
        l1_whitened = l1_whitened_tensor.squeeze().cpu().numpy()
        #print(f"[DEBUG whiten_data] l1_whitened.shape (final numpy)={l1_whitened.shape}")

    #print(f"[DEBUG whiten_data] Returning h1_whitened={h1_whitened.shape}, l1_whitened={l1_whitened.shape}\n")
    return h1_whitened, l1_whitened


# ============================================================================
# cWB Metric Computation (from extract_cwb_features_...)
# ============================================================================

def _fft_of_already_whitened(x, fs, nfft=None, window_time=True):
    """Compute FFT of already whitened data."""
    if nfft is None:
        nfft = len(x)
    w = np.hanning(len(x)) if window_time else np.ones(len(x))
    X = rfft(w * x, n=nfft)
    f = rfftfreq(nfft, 1.0 / fs)
    win_rms = np.sqrt(np.mean(w ** 2))
    norm = np.sqrt(0.5 * fs * nfft) * win_rms
    S = X / norm
    return f, S


def _compute_whitened_pair(xH, xL, fs, *, nperseg, noverlap, nfft, window_time):
    """Compute whitened frequency-domain representation."""
    fH, SH = _fft_of_already_whitened(xH, fs, nfft=nfft, window_time=window_time)
    fL, SL = _fft_of_already_whitened(xL, fs, nfft=nfft, window_time=window_time)

    if len(fH) != len(fL) or not np.allclose(fH, fL):
        raise ValueError("Frequency grids differ")

    S = np.stack([SH, SL], axis=0)
    return fH, S


def _restrict_band(f, S, flo, fhi):
    """Restrict to frequency band."""
    mask = (f >= flo) & (f <= fhi)
    if not np.any(mask):
        return None, None
    return f[mask], S[:, mask]


def _phase_for_delays(f, delays, n_detectors):
    """Compute phase shifts for time delays."""
    delays = np.asarray(delays, dtype=float)
    if delays.size == 1:
        delays = np.repeat(delays, n_detectors)
    if delays.size != n_detectors:
        raise ValueError(f"Expected {n_detectors} delays, got {delays.size}")
    return np.exp(1j * 2 * np.pi * f[None, :] * delays[:, None])


def _make_projector(K, F=None):
    """Create projection operator."""
    if F is None:
        u = np.ones((K, 1), dtype=float)
        u /= np.linalg.norm(u)
        P = u @ u.T
        return P.astype(complex)
    A = np.asarray(F)
    if A.ndim == 1:
        A = A[:, None]
    A = A.astype(complex)
    G = A.conj().T @ A
    G += 1e-12 * np.eye(G.shape[0], dtype=complex)
    P = A @ np.linalg.inv(G) @ A.conj().T
    P = 0.5 * (P + P.conj().T)
    return P


def _cwb_stats_core(S, P, I):
    """Compute cWB statistics."""
    if S.size == 0:
        return dict(cc=np.nan, rho=np.nan, scc=np.nan, edr=np.nan)

    S_sig = P @ S
    S_res = (I - P) @ S

    abs_S_sq = np.abs(S) ** 2
    E = float(np.sum(abs_S_sq))
    Em = float(np.sum(np.max(abs_S_sq, axis=0)))

    S_sig_sq = np.abs(S_sig) ** 2
    L = float(np.sum(np.sum(S_sig_sq, axis=0)))

    cross = np.sum(S, axis=0)
    coherent_per_bin = np.abs(cross) ** 2 - np.sum(abs_S_sq, axis=0)
    C = float(np.sum(np.real(coherent_per_bin)))

    N0 = float(np.sum(np.abs(S_res) ** 2))

    Cpos = max(C, 0.0)
    denom_cc = abs(C) + N0 + EPS
    cc = C / denom_cc if denom_cc > EPS else 0.0
    rho = np.sqrt(Cpos / max(S.shape[0] - 1, 1)) if S.shape[0] > 1 else float("nan")

    n_pix = S.shape[1]
    denom_scc = (E - Em) + (E - L) + n_pix
    scc = (E - Em) / denom_scc if denom_scc > EPS else 0.0
    edr = N0 / max(Cpos, EPS)

    return dict(cc=float(cc), rho=float(rho), scc=float(scc), edr=float(edr))


def _cwb_stats_from_band(S_band, f_band, delays, P, I):
    """Compute cWB stats with delays."""
    if S_band is None or f_band is None:
        return dict(cc=np.nan, rho=np.nan, scc=np.nan, edr=np.nan)
    phase = _phase_for_delays(f_band, delays, S_band.shape[0])
    S_shifted = S_band * phase
    return _cwb_stats_core(S_shifted, P, I)


def _cwb_scan_over_delays(S_band, f_band, max_tau, n_tau, P, I):
    """Scan over delays to maximize cc."""
    if S_band is None or f_band is None:
        return (np.nan, np.nan, np.nan, np.nan)

    taus = np.linspace(-max_tau, +max_tau, int(n_tau))
    best_cc = -np.inf
    best_vals = (np.nan, np.nan, np.nan, np.nan)

    for tau in taus:
        stats = _cwb_stats_from_band(S_band, f_band, delays=(0.0, tau), P=P, I=I)
        cc = stats["cc"]
        if np.isfinite(cc) and cc > best_cc:
            best_cc = cc
            best_vals = (stats["cc"], stats["rho"], stats["scc"], stats["edr"])

    return best_vals


def _freq_corr_band(xH, xL, fs, flo, fhi, nperseg=256, noverlap=128,
                    nfft=4096, window_time=True):
    """Frequency-domain correlation."""
    f, S = _compute_whitened_pair(xH, xL, fs, nperseg=nperseg, noverlap=noverlap,
                                   nfft=nfft, window_time=window_time)
    f_band, S_band = _restrict_band(f, S, flo, fhi)

    if S_band is None or f_band is None:
        return np.nan, np.nan

    SH = S_band[0, :]
    SL = S_band[1, :]
    num = np.sum(SH * np.conj(SL))
    den = np.sqrt(np.sum(np.abs(SH)**2) * np.sum(np.abs(SL)**2)) + EPS
    z = num / den
    return float(np.abs(z)), float(np.real(z))


def _coherence_simple(xH, xL, fs, flo, fhi, nperseg=256, noverlap=128):
    """Coherence metrics."""
    f, C = coherence(xH, xL, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if f is None or C is None or len(f) == 0:
        return np.nan, np.nan
    mask = (f >= flo) & (f <= fhi)
    if not np.any(mask):
        return np.nan, np.nan
    Cb = C[mask]
    return float(np.nanmean(Cb)), float(np.nanmax(Cb))


def compute_all_metrics(h1_whitened, l1_whitened, fs=SAMPLE_RATE,
                       band=BAND, use_delay_scan=USE_DELAY_SCAN):
    """Compute all 8 key cWB metrics."""
    flo, fhi = band

    # Prepare frequency-domain data
    f, S = _compute_whitened_pair(h1_whitened, l1_whitened, fs,
                                   nperseg=NPERSEG, noverlap=NOVERLAP,
                                   nfft=NFFT, window_time=True)
    f_band, S_band = _restrict_band(f, S, flo, fhi)

    # Projectors
    P = _make_projector(2, F=None)
    I = np.eye(2, dtype=complex)

    # cWB stats
    if use_delay_scan:
       cc, rho, scc, edr = _cwb_scan_over_delays(S_band, f_band, TAU_MAX, N_TAU, P, I)
    else:
       stats = _cwb_stats_from_band(S_band, f_band, delays=(0.0, 0.0), P=P, I=I)
       cc, rho, scc, edr = stats["cc"], stats["rho"], stats["scc"], stats["edr"]

    # Coherence
    coh_mean, coh_max = _coherence_simple(h1_whitened, l1_whitened, fs, flo, fhi,
                                          nperseg=NPERSEG, noverlap=NOVERLAP)

    # Frequency-domain correlation
    corr_mag_fd, corr_real_fd = _freq_corr_band(h1_whitened, l1_whitened, fs, flo, fhi,
                                                nperseg=NPERSEG, noverlap=NOVERLAP,
                                                nfft=NFFT, window_time=True)

    return {
        "rho": float(rho),
        "edr": float(edr),
        "cc": float(cc),
        "scc": float(scc),
        "coh_max": float(coh_max),
        "coh_mean": float(coh_mean),
        "corr_mag_fd": float(corr_mag_fd),
        "corr_real_fd": float(corr_real_fd),
    }


# ============================================================================
# Data Loading (from read_strain_data.py)
# ============================================================================

def load_segment_data(error_config_path, data_dir, segment_idx):
    """
    Load strain data for a segment.

    Parameters
    ----------
    error_config_path : Path
        Path to error_config.h5
    data_dir : Path
        Directory with strain data files
    segment_idx : int
        Index of segment

    Returns
    -------
    h1_strain, l1_strain, metadata
    """
    with h5py.File(error_config_path, "r") as f:
        error_data = f["data"][:]

    err_info = error_data[segment_idx]

    t0 = int(err_info[0])
    length = int(err_info[1])
    shift = int(err_info[2])
    start = err_info[3]
    end = err_info[4]
    dur = err_info[5]
    gwak_value = err_info[6]

    # Data file path
    data_file = Path(data_dir) / f"background-{t0}-{length}.h5"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Calculate indices (from read_strain_data.py)
    error_time = start - t0
    idx = error_time - PSD_LENGTH - FFT_LENGTH - KERNEL / 2

    idx_start = int(idx * SAMPLE_RATE)
    idx_end = int((idx + PSD_LENGTH + KERNEL + 2 * FFT_LENGTH) * SAMPLE_RATE)

    # Read strain
    with h5py.File(data_file, "r") as f:
        h1_strain = f["H1"][idx_start:idx_end]
        l1_strain = f["L1"][idx_start + shift * SAMPLE_RATE:
                           idx_end + shift * SAMPLE_RATE]

    metadata = {
        "segment_idx": segment_idx,
        "t0": t0,
        "length": length,
        "shift": shift,
        "error_start": float(start),
        "error_end": float(end),
        "duration": float(dur),
        "gwak_value": float(gwak_value),
    }

    return h1_strain, l1_strain, metadata


# ============================================================================
# Veto Application
# ============================================================================

def apply_vetoes(metrics, thresholds):
    """Apply veto thresholds to metrics."""
    vetoed = False
    failed_cuts = []

    for metric_name in ["rho", "edr", "cc", "scc", 
                        "coh_max", "coh_mean",
                       "corr_mag_fd", "corr_real_fd"]:
        metric_value = metrics[metric_name]
        thresh = thresholds[metric_name]

        # Check min threshold
        if thresh["min"] is not None:
            if not np.isfinite(metric_value) or metric_value < thresh["min"]:
                vetoed = True
                failed_cuts.append(f"{metric_name} < {thresh['min']}")

        # Check max threshold
        if thresh["max"] is not None:
            if not np.isfinite(metric_value) or metric_value > thresh["max"]:
                vetoed = True
                failed_cuts.append(f"{metric_name} > {thresh['max']}")

    return {
        "vetoed": vetoed,
        "failed_cuts": failed_cuts,
        "n_failed_cuts": len(failed_cuts),
    }


# ============================================================================
# Main Processing
# ============================================================================

def process_segment(error_config_path, data_dir, segment_idx, thresholds):
    """Process a single segment."""
    try:
        # Load data
        h1_strain, l1_strain, metadata = load_segment_data(
            error_config_path, data_dir, segment_idx
        )

        # Whiten
        h1_whitened, l1_whitened = whiten_data(h1_strain, l1_strain)

        # Compute metrics
        metrics = compute_all_metrics(h1_whitened, l1_whitened)

        # Apply vetoes
        veto_info = apply_vetoes(metrics, thresholds)

        result = {
            "success": True,
            "error": None,
            **metadata,
            **metrics,
            **veto_info,
        }

    except Exception as e:
        result = {
            "segment_idx": segment_idx,
            "success": False,
            "error": str(e),
            "vetoed": None,
        }

    return result


def main():
    global DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--error-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Set device
    DEVICE = args.device

    # Load thresholds
    with open(args.thresholds, "r") as f:
        thresholds = json.load(f)

    # Get number of segments
    with h5py.File(args.error_config, "r") as f:
        n_segments = len(f["data"][:])

    if args.first_n is not None:
        segment_indices = range(min(args.first_n, n_segments))
    else:
        segment_indices = range(n_segments)

    print(f"Processing {len(segment_indices)} segments...")
    print(f"Device: {DEVICE}")

    results = []
    for idx in tqdm(segment_indices):
        result = process_segment(args.error_config, args.data_dir, idx, thresholds)
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    # Print summary
    n_success = df["success"].sum()
    print(f"\nSuccessfully processed: {n_success}/{len(df)}")

    if n_success > 0:
        success_df = df[df["success"]]
        n_vetoed = success_df["vetoed"].sum()
        print(f"Vetoed: {n_vetoed}/{n_success}")
        print(f"Passed: {n_success - n_vetoed}/{n_success}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
