# cwb_stats.py
import math
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import welch

def _fft_of_already_whitened(x, fs, nfft=None, window_time=True):
    if nfft is None:
        nfft = len(x)
    w = np.hanning(len(x)) if window_time else np.ones(len(x))
    X = rfft(w * x, n=nfft)
    f = rfftfreq(nfft, 1.0/fs)
    win_rms = np.sqrt(np.mean(w**2))
    S = X / (np.sqrt(0.5 * fs * nfft) * win_rms)
    return f, S

def _make_projector(K, F=None):
    if F is None:
        u = np.ones((K, 1), dtype=float); u /= np.linalg.norm(u)
        return (u @ u.T).astype(complex)
    A = np.asarray(F);  A = A[:, None] if A.ndim == 1 else A
    A = A.astype(complex)
    G = A.conj().T @ A + 1e-12 * np.eye(A.shape[1], dtype=complex)
    P = A @ np.linalg.inv(G) @ A.conj().T
    return 0.5 * (P + P.conj().T)

def cwb_stats_2ifo(
    xH, xL, fs, flo, fhi, *,
    nperseg=256, noverlap=128, nfft=4096, window_time=True, delays=(0.0, 0.0), F=None
):
    fH, SH = _fft_of_already_whitened(xH, fs, nfft=nfft, window_time=window_time)
    fL, SL = _fft_of_already_whitened(xL, fs, nfft=nfft, window_time=window_time)
    if len(fH) != len(fL) or not np.allclose(fH, fL):
        raise ValueError("Mismatched frequency grids.")
    f = fH

    # 2) apply relative delays
    delays = np.asarray(delays, float)
    phase = np.exp(1j * 2*np.pi * f[None, :] * delays[:, None])
    S = np.stack([SH, SL], axis=0) * phase

    # 3) band limit
    band = (f >= flo) & (f <= fhi)
    if not np.any(band):
        return dict(cc=np.nan, rho=np.nan, scc=np.nan, edr=np.nan,
                    C=np.nan, N=np.nan, E=np.nan, Em=np.nan, L=np.nan, K=2, nbins=0)
    S = S[:, band]
    K, M = S.shape[0], S.shape[1]
    I = np.eye(K, dtype=complex)
    P = _make_projector(K, F=F)

    # 4) signal / residual
    S_sig = P @ S
    S_res = (I - P) @ S

    # 5) energies
    E = float(np.sum(np.abs(S)**2))
    per_det_en = np.abs(S)**2
    Em = float(np.sum(np.max(per_det_en, axis=0)))
    L_energy_bins = np.sum(np.abs(S_sig)**2, axis=0)
    L = float(np.sum(L_energy_bins))
    sum_sig = np.sum(S_sig, axis=0)
    offdiag_per_bin = np.abs(sum_sig)**2 - L_energy_bins
    C = float(np.sum(np.real(offdiag_per_bin)))
    N0 = float(np.sum(np.abs(S_res)**2))

    # 6) stats
    Cpos = max(C, 0.0)
    cc  = C / (abs(C) + N0 + 1e-30)
    rho = math.sqrt(Cpos / (K - 1)) if (K > 1) else float('nan')
    n_pix = M
    denom_scc = (E - Em) + (E - L) + n_pix
    scc = (E - Em) / denom_scc if denom_scc > 0 else 0.0
    edr = (N0 / Cpos) if Cpos > 0 else float('inf')

    return dict(cc=float(cc), rho=float(rho), scc=float(scc), edr=float(edr),
                C=C, N=N0, E=E, Em=Em, L=L, K=K, nbins=M)

def cwb_cc_rho_max_over_delay_2ifo(
    xH, xL, fs, flo, fhi, *, max_tau=0.010, n_tau=81,
    nperseg=256, noverlap=128, nfft=4096, window_time=True, F=None
):
    taus = np.linspace(-max_tau, +max_tau, int(n_tau))
    best_cc = -np.inf
    best = (np.nan, np.nan, np.nan, np.nan)
    for tau in taus:
        st = cwb_stats_2ifo(
            xH, xL, fs, flo, fhi, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            window_time=window_time, delays=(0.0, tau), F=F
        )
        cci = st["cc"]
        if np.isfinite(cci) and cci > best_cc:
            best_cc = cci
            best = (st["cc"], st["rho"], st["scc"], st["edr"])
    return best