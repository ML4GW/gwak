# cwb_stats.py
import math
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import welch, coherence, correlate

EPS = 1e-12

FS = 4096.0
BAND = (32.0, 2048.0)

N_MAX_PER_CLASS = 300
BATCH_SIZE = 1024

NPERSEG = 256
NOVERLAP = 128
NFFT = 4096

USE_PREWHITENED = True
WINDOW_TIME = True

USE_DELAY_SCAN = True
TAU_MAX = 0.010
N_TAU = 81

SHUFFLE_EXAMPLES = True
SHUFFLE_SEED = 12345

OUT_DIR = "features_with_gaussianity_crosscorr"


def _fft_of_already_whitened(x, fs, nfft=None, window_time=True):
    if nfft is None:
        nfft = len(x)
    w = np.hanning(len(x)) if window_time else np.ones(len(x))
    X = rfft(w * x, n=nfft)
    f = rfftfreq(nfft, 1.0 / fs)
    win_rms = np.sqrt(np.mean(w ** 2))
    norm = np.sqrt(0.5 * fs * nfft) * win_rms
    S = X / norm
    return f, S


def _whitened_fft_via_psd(x, fs, f_psd, Pxx, nfft=None, window_time=True):
    if nfft is None:
        nfft = len(x)
    w = np.hanning(len(x)) if window_time else np.ones(len(x))
    X = rfft(w * x, n=nfft)
    f = rfftfreq(nfft, 1.0 / fs)
    P = np.interp(f, f_psd, Pxx, left=Pxx[0], right=Pxx[-1]) + 1e-30
    win_rms = np.sqrt(np.mean(w ** 2))
    norm = np.sqrt(0.5 * fs * nfft) * win_rms
    S = X / np.sqrt(P) / norm
    return f, S


def _compute_whitened_pair(xH, xL, fs, *, nperseg, noverlap, nfft, window_time, use_prewhitened):
    if use_prewhitened:
        fH, SH = _fft_of_already_whitened(xH, fs, nfft=nfft, window_time=window_time)
        fL, SL = _fft_of_already_whitened(xL, fs, nfft=nfft, window_time=window_time)
    else:
        fH_psd, PH = welch(xH, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann", detrend="constant")
        fL_psd, PL = welch(xL, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann", detrend="constant")
        fH, SH = _whitened_fft_via_psd(xH, fs, fH_psd, PH, nfft=nfft, window_time=window_time)
        fL, SL = _whitened_fft_via_psd(xL, fs, fL_psd, PL, nfft=nfft, window_time=window_time)

    if len(fH) != len(fL) or not np.allclose(fH, fL):
        raise ValueError("Frequency grids differ; ensure same fs and nfft for both IFOs.")
    S = np.stack([SH, SL], axis=0)
    return fH, S


def _restrict_band(f, S, flo, fhi):
    mask = (f >= flo) & (f <= fhi)
    if not np.any(mask):
        return None, None
    return f[mask], S[:, mask]


def _phase_for_delays(f, delays, n_detectors):
    delays = np.asarray(delays, dtype=float)
    if delays.size == 1:
        delays = np.repeat(delays, n_detectors)
    if delays.size != n_detectors:
        raise ValueError(f"Expected {n_detectors} delays, got {delays.size}")
    return np.exp(1j * 2 * np.pi * f[None, :] * delays[:, None])


def _apply_fractional_delay_time(x, fs, tau):
    """
    Fractional time shift using frequency-domain phase and IFFT.
    x: 1D array, fs: sampling rate [Hz], tau: seconds (positive => advance).
    Returns a real-valued time series aligned with original length.
    """
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.rfft(x, n=n)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    phase = np.exp(1j * 2.0 * np.pi * f * tau)
    Xs = X * phase
    xs = np.fft.irfft(Xs, n=n)
    return xs


def _make_projector(K, F=None):
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
    """Compute cWB-style stats; coherent energy uses raw S (no projection)."""
    if S.size == 0:
        return dict(cc=np.nan, rho=np.nan, scc=np.nan, edr=np.nan,
                    C=np.nan, N=np.nan, E=np.nan, Em=np.nan, L=np.nan,
                    K=S.shape[0], nbins=0)

    # Projected components for L/N and other energies
    S_sig = P @ S
    S_res = (I - P) @ S

    abs_S_sq = np.abs(S) ** 2
    E = float(np.sum(abs_S_sq))
    Em = float(np.sum(np.max(abs_S_sq, axis=0)))

    S_sig_sq = np.abs(S_sig) ** 2
    L = float(np.sum(np.sum(S_sig_sq, axis=0)))

    # Coherent energy using raw detector streams (allows negative contributions)
    cross = np.sum(S, axis=0)
    coherent_per_bin = np.abs(cross) ** 2 - np.sum(abs_S_sq, axis=0)
    C = float(np.sum(np.real(coherent_per_bin)))

    N0 = float(np.sum(np.abs(S_res) ** 2))

    Cpos = max(C, 0.0)
    denom_cc = abs(C) + N0 + EPS
    cc = C / denom_cc if denom_cc > EPS else 0.0
    rho = math.sqrt(Cpos / max(S.shape[0] - 1, 1)) if S.shape[0] > 1 else float("nan")

    n_pix = S.shape[1]
    denom_scc = (E - Em) + (E - L) + n_pix
    scc = (E - Em) / denom_scc if denom_scc > EPS else 0.0
    edr = N0 / max(Cpos, EPS)

    return dict(cc=float(cc), rho=float(rho), scc=float(scc), edr=float(edr),
                C=C, N=N0, E=E, Em=Em, L=L, K=S.shape[0], nbins=n_pix)


def _cwb_stats_from_band(S_band, f_band, delays, P, I):
    if S_band is None or f_band is None:
        return dict(cc=np.nan, rho=np.nan, scc=np.nan, edr=np.nan,
                    C=np.nan, N=np.nan, E=np.nan, Em=np.nan, L=np.nan,
                    K=0, nbins=0)
    phase = _phase_for_delays(f_band, delays, S_band.shape[0])
    S_shifted = S_band * phase
    return _cwb_stats_core(S_shifted, P, I)


def _prepare_bandlimited_pair(xH, xL, fs, flo, fhi, *, nperseg, noverlap, nfft,
                              window_time, use_prewhitened):
    f, S = _compute_whitened_pair(
        xH, xL, fs,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window_time=window_time, use_prewhitened=use_prewhitened
    )
    return _restrict_band(f, S, flo, fhi)


# -----------------------------
# Simple correlation helpers
# -----------------------------
def _pearsonr_safe(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n = min(x.size, y.size)
    if n < 2:
        return np.nan
    if x.size != y.size:
        x = x[:n]; y = y[:n]
    sx = np.std(x); sy = np.std(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _xcorr_metrics(x, y):
    """
    Normalized cross-correlation using scipy.signal.correlate (FFT-based).
    Returns: max_abs_corr, lag_at_max_samples, corr_at_zero_lag
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - np.nanmean(x); y = y - np.nanmean(y)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n = min(x.size, y.size)
    if n < 2:
        return np.nan, 0, np.nan
    if x.size != y.size:
        x = x[:n]; y = y[:n]
    denom = np.sqrt(np.sum(x*x) * np.sum(y*y))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan, 0, np.nan
    xc = correlate(x, y, mode='full', method='fft') / denom
    lags = np.arange(-n+1, n)
    idx = int(np.nanargmax(np.abs(xc)))
    max_corr = float(xc[idx])
    lag_at_max = int(lags[idx])
    zero_idx = n - 1
    corr0 = float(xc[zero_idx])
    return max_corr, lag_at_max, corr0


def _freq_corr_band(xH, xL, fs, flo, fhi, use_prewhitened=True, nperseg=256, noverlap=128, nfft=4096, window_time=True, tau=0.0):
    """
    Frequency-domain normalized correlation over [flo, fhi].
    Uses the same whitening/bandlimiting as the cWB features.
    Returns: mag, real, imag, phase_rad
    """
    f_band, S_band = _prepare_bandlimited_pair(
        xH, xL, fs, flo, fhi,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window_time=window_time, use_prewhitened=use_prewhitened
    )
    if S_band is None or f_band is None:
        return np.nan, np.nan, np.nan, np.nan
    phase = _phase_for_delays(f_band, (0.0, tau), S_band.shape[0])
    S_shift = S_band * phase
    SH = S_shift[0, :]; SL = S_shift[1, :]
    num = np.sum(SH * np.conj(SL))
    den = np.sqrt(np.sum(np.abs(SH)**2) * np.sum(np.abs(SL)**2)) + EPS
    z = num / den
    return float(np.abs(z)), float(np.real(z)), float(np.imag(z)), float(np.angle(z))


def _coherence_simple(xH, xL, fs, flo, fhi, nperseg=256, noverlap=128):
    """
    SciPy magnitude-squared coherence averaged/maxed in band.
    """
    f, C = coherence(xH, xL, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if f is None or C is None or len(f) == 0:
        return np.nan, np.nan
    mask = (f >= flo) & (f <= fhi)
    if not np.any(mask):
        return np.nan, np.nan
    Cb = C[mask]
    return float(np.nanmean(Cb)), float(np.nanmax(Cb))


def _cwb_scan_over_delays(S_band, f_band, max_tau, n_tau, P, I):
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


def _cwb_scan_over_delays_with_tau(S_band, f_band, max_tau, n_tau, P, I):
    if S_band is None or f_band is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    taus = np.linspace(-max_tau, +max_tau, int(n_tau))
    best_cc = -np.inf
    best_vals = (np.nan, np.nan, np.nan, np.nan, 0.0)
    for tau in taus:
        stats = _cwb_stats_from_band(S_band, f_band, delays=(0.0, tau), P=P, I=I)
        cc = stats["cc"]
        if np.isfinite(cc) and cc > best_cc:
            best_cc = cc
            best_vals = (stats["cc"], stats["rho"], stats["scc"], stats["edr"], float(tau))
    return best_vals


def cwb_stats_and_delay_2ifo(xH, xL, fs, flo, fhi, *, max_tau=0.010, n_tau=81,
                             nperseg=256, noverlap=128, nfft=4096,
                             window_time=True, use_prewhitened=True, F=None):
    f_band, S_band = _prepare_bandlimited_pair(
        xH, xL, fs, flo, fhi,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window_time=window_time, use_prewhitened=use_prewhitened
    )
    if S_band is None:
        return (np.nan, np.nan, np.nan, np.nan, 0.0)
    K = S_band.shape[0]
    I = np.eye(K, dtype=complex)
    P = _make_projector(K, F=F)
    return _cwb_scan_over_delays_with_tau(S_band, f_band, max_tau, n_tau, P, I)


def cwb_stats_2ifo(xH, xL, fs, flo, fhi, *, nperseg=256, noverlap=128, nfft=4096,
                   window_time=True, use_prewhitened=True, delays=(0.0, 0.0), F=None):
    f_band, S_band = _prepare_bandlimited_pair(
        xH, xL, fs, flo, fhi,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window_time=window_time, use_prewhitened=use_prewhitened
    )
    if S_band is None:
        return dict(cc=np.nan, rho=np.nan, scc=np.nan, edr=np.nan, C=np.nan, N=np.nan,
                    E=np.nan, Em=np.nan, L=np.nan, K=2, nbins=0)

    K = S_band.shape[0]
    I = np.eye(K, dtype=complex)
    P = _make_projector(K, F=F)
    return _cwb_stats_from_band(S_band, f_band, delays, P, I)


def cwb_cc_rho_max_over_delay_2ifo(xH, xL, fs, flo, fhi, *, max_tau=0.010, n_tau=81,
                                   nperseg=256, noverlap=128, nfft=4096,
                                   window_time=True, use_prewhitened=True, F=None):
    f_band, S_band = _prepare_bandlimited_pair(
        xH, xL, fs, flo, fhi,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window_time=window_time, use_prewhitened=use_prewhitened
    )
    if S_band is None:
        return (np.nan, np.nan, np.nan, np.nan)

    K = S_band.shape[0]
    I = np.eye(K, dtype=complex)
    P = _make_projector(K, F=F)
    return _cwb_scan_over_delays(S_band, f_band, max_tau, n_tau, P, I)

# def _fft_of_already_whitened(x, fs, nfft=None, window_time=True):
#     if nfft is None:
#         nfft = len(x)
#     w = np.hanning(len(x)) if window_time else np.ones(len(x))
#     X = rfft(w * x, n=nfft)
#     f = rfftfreq(nfft, 1.0/fs)
#     win_rms = np.sqrt(np.mean(w**2))
#     S = X / (np.sqrt(0.5 * fs * nfft) * win_rms)
#     return f, S

# def _make_projector(K, F=None):
#     if F is None:
#         u = np.ones((K, 1), dtype=float); u /= np.linalg.norm(u)
#         return (u @ u.T).astype(complex)
#     A = np.asarray(F);  A = A[:, None] if A.ndim == 1 else A
#     A = A.astype(complex)
#     G = A.conj().T @ A + 1e-12 * np.eye(A.shape[1], dtype=complex)
#     P = A @ np.linalg.inv(G) @ A.conj().T
#     return 0.5 * (P + P.conj().T)

# def cwb_stats_2ifo(
#     xH, xL, fs, flo, fhi, *,
#     nperseg=256, noverlap=128, nfft=4096, window_time=True, delays=(0.0, 0.0), F=None
# ):
#     fH, SH = _fft_of_already_whitened(xH, fs, nfft=nfft, window_time=window_time)
#     fL, SL = _fft_of_already_whitened(xL, fs, nfft=nfft, window_time=window_time)
#     if len(fH) != len(fL) or not np.allclose(fH, fL):
#         raise ValueError("Mismatched frequency grids.")
#     f = fH

#     # 2) apply relative delays
#     delays = np.asarray(delays, float)
#     phase = np.exp(1j * 2*np.pi * f[None, :] * delays[:, None])
#     S = np.stack([SH, SL], axis=0) * phase

#     # 3) band limit
#     band = (f >= flo) & (f <= fhi)
#     if not np.any(band):
#         return dict(cc=np.nan, rho=np.nan, scc=np.nan, edr=np.nan,
#                     C=np.nan, N=np.nan, E=np.nan, Em=np.nan, L=np.nan, K=2, nbins=0)
#     S = S[:, band]
#     K, M = S.shape[0], S.shape[1]
#     I = np.eye(K, dtype=complex)
#     P = _make_projector(K, F=F)

#     # 4) signal / residual
#     S_sig = P @ S
#     S_res = (I - P) @ S

#     # 5) energies
#     E = float(np.sum(np.abs(S)**2))
#     per_det_en = np.abs(S)**2
#     Em = float(np.sum(np.max(per_det_en, axis=0)))
#     L_energy_bins = np.sum(np.abs(S_sig)**2, axis=0)
#     L = float(np.sum(L_energy_bins))
#     sum_sig = np.sum(S_sig, axis=0)
#     offdiag_per_bin = np.abs(sum_sig)**2 - L_energy_bins
#     C = float(np.sum(np.real(offdiag_per_bin)))
#     N0 = float(np.sum(np.abs(S_res)**2))

#     # 6) stats
#     Cpos = max(C, 0.0)
#     cc  = C / (abs(C) + N0 + 1e-30)
#     rho = math.sqrt(Cpos / (K - 1)) if (K > 1) else float('nan')
#     n_pix = M
#     denom_scc = (E - Em) + (E - L) + n_pix
#     scc = (E - Em) / denom_scc if denom_scc > 0 else 0.0
#     edr = (N0 / Cpos) if Cpos > 0 else float('inf')

#     return dict(cc=float(cc), rho=float(rho), scc=float(scc), edr=float(edr),
#                 C=C, N=N0, E=E, Em=Em, L=L, K=K, nbins=M)

# def cwb_cc_rho_max_over_delay_2ifo(
#     xH, xL, fs, flo, fhi, *, max_tau=0.010, n_tau=81,
#     nperseg=256, noverlap=128, nfft=4096, window_time=True, F=None
# ):
#     taus = np.linspace(-max_tau, +max_tau, int(n_tau))
#     best_cc = -np.inf
#     best = (np.nan, np.nan, np.nan, np.nan)
#     for tau in taus:
#         st = cwb_stats_2ifo(
#             xH, xL, fs, flo, fhi, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
#             window_time=window_time, delays=(0.0, tau), F=F
#         )
#         cci = st["cc"]
#         if np.isfinite(cci) and cci > best_cc:
#             best_cc = cci
#             best = (st["cc"], st["rho"], st["scc"], st["edr"])
#     return best

# # ------ helpers for extra postselection variables (inspired by reference script) -----

# def _freq_corr_band(xH, xL, fs, flo, fhi, use_prewhitened=True, nperseg=256, noverlap=128, nfft=4096, window_time=True, tau=0.0):
#     """
#     Frequency-domain normalized correlation over [flo, fhi].
#     Uses the same whitening/bandlimiting as the cWB features.
#     Returns: mag, real, imag, phase_rad
#     """
#     f_band, S_band = _prepare_bandlimited_pair(
#         xH, xL, fs, flo, fhi,
#         nperseg=nperseg, noverlap=noverlap, nfft=nfft,
#         window_time=window_time, use_prewhitened=use_prewhitened
#     )
#     if S_band is None or f_band is None:
#         return np.nan, np.nan, np.nan, np.nan
#     phase = _phase_for_delays(f_band, (0.0, tau), S_band.shape[0])
#     S_shift = S_band * phase
#     SH = S_shift[0, :]; SL = S_shift[1, :]
#     num = np.sum(SH * np.conj(SL))
#     den = np.sqrt(np.sum(np.abs(SH)**2) * np.sum(np.abs(SL)**2)) + EPS
#     z = num / den
#     return float(np.abs(z)), float(np.real(z)), float(np.imag(z)), float(np.angle(z))



# def _coherence_simple(xH, xL, fs, flo, fhi, nperseg=256, noverlap=128):
#     """
#     SciPy magnitude-squared coherence averaged/maxed in band.
#     Returns: coh_mean, coh_max
#     """
#     xH = np.asarray(xH, float)
#     xL = np.asarray(xL, float)
#     f, C = coherence(xH, xL, fs=fs, nperseg=nperseg, noverlap=noverlap)
#     if f is None or C is None or len(f) == 0:
#         return np.nan, np.nan
#     mask = (f >= flo) & (f <= fhi)
#     if not np.any(mask):
#         return np.nan, np.nan
#     Cb = C[mask]
#     return float(np.nanmean(Cb)), float(np.nanmax(Cb))