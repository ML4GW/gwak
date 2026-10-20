"""H1-L1 correlation features for GWAK (torch, traceable with torch.jit.trace).

Input  : (B, 2, 4096) whitened 1 s kernels at 4096 Hz, index 0 = H1, 1 = L1, any dtype / overall scale
         (each event and detector is standardised inside). Kernels must not contain NaN.
Output : (B, F) float32 in the order of FEATURES[mode]; mode "corr21" (default, F = 21) or "corr15" (F = 15).

  time domain (9)   pearson_0lag (abs), cc_maxlag_1s, cc_maxlag_win512, cc_band_maxz, msc_max,
                    xc_env_w64, xc_env_w128, xc_env_w256, xc_env_w256_lag_ms (abs)
  TF coherence (6)  ccen_tf, ccen_tf_reg, wcc2_tf, balance, dt_peak_ms, dlogf_peak
  TF loudness (6)   ecoh_tf, enet_peak, epeak_H, epeak_L, e_clu_H, e_clu_L        (corr21 only)

Torch port of GWAK_Heuristics/corrcuts_v2 (validated on BBC injections + the one-month whitened timeslides):
time domain = features_v2.time_domain + envelope_window (numpy, float64); TF = trusted/ml4gw_tf.py (multi-Q
coherence on ml4gw QTile rows, Q = 4, 11, 32, 90, 256). Differences from ml4gw_tf.py, none changing eager results:
  * all tables are registered buffers; no numpy in forward;
  * the nearest-tile lag shift round(tau * n_row / 4096) is a precomputed integer table (half-to-even, as eager
    torch.round). torch.round inside a traced graph is compiled by the TorchScript fuser with a different rule for
    exact halves, which changed ccen_tf / wcc2_tf in ~24% of kernels from the second traced call on;
  * flat (all-zero) kernels give finite output instead of NaN.
Tables are built once at construction with ml4gw (QTile, SingleQTransform.get_freqs); forward uses torch only.
"""
import math
import types

import numpy as np
import torch
from torch import nn

SR = 4096
N = 4096
MAXLAG = 44                                     # samples = 10.7 ms
LAGS = np.arange(-MAXLAG, MAXLAG + 1)

TD_FEATURES = ["pearson_0lag", "cc_maxlag_1s", "cc_maxlag_win512", "cc_band_maxz", "msc_max",
               "xc_env_w64", "xc_env_w128", "xc_env_w256", "xc_env_w256_lag_ms"]
TF_COH_FEATURES = ["ccen_tf", "ccen_tf_reg", "wcc2_tf", "balance", "dt_peak_ms", "dlogf_peak"]
TF_LOUD_FEATURES = ["ecoh_tf", "enet_peak", "epeak_H", "epeak_L", "e_clu_H", "e_clu_L"]
FEATURES = {
    "corr15": TD_FEATURES + TF_COH_FEATURES,
    "corr21": TD_FEATURES + TF_COH_FEATURES + TF_LOUD_FEATURES,
}

# time domain
BANDS = [(20, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2000)]
ENV_BAND = (30.0, 1900.0)
ENV_SMOOTH = 33                                 # ~8 ms boxcar on the envelope
ENV_HALF = (64, 128, 256)                       # window half-widths, samples
WIN, HOP = 512, 256                             # short-window NCC and Welch coherence segments
MSC_FRANGE = (20.0, 1024.0)

# time-frequency
QS = (4.0, 11.0, 32.0, 90.0, 256.0)
E_THR, N_MIN, N_MAX, DEAD = 12.0, 16, 4096, 0.1
LINE_HZ = {32.0: (58.0, 59.0), 90.0: (59.0,)}  # line rows on the one-month whitened timeslides
ALPHAS = (2.191e-07, 2.345e-07, 2.348e-07, 2.334e-07, 2.314e-07)   # tile weight per plane: sum w e = SNR^2


def _band_gain(lo, hi):
    """Zero-phase FFT bandpass gain with 4 Hz cosine edges (compute_corr_features.bandpass)."""
    f = np.fft.rfftfreq(N, 1 / SR)
    g = np.clip((f - (lo - 4)) / 8, 0, 1) * np.clip(((hi + 4) - f) / 8, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * g)


def _standardise(x):
    x = x - x.mean(-1, keepdim=True)
    s = x.std(-1, keepdim=True)
    return x / torch.where(s > 0, s, torch.ones_like(s))


def _lagged_ncc(a, b, width):
    """rho(tau) = sum a(t) b(t + tau) / sqrt(Ea Eb), tau = -MAXLAG..MAXLAG, means removed. (..., width) -> (..., 89)"""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    nfft = 1 << int(math.ceil(math.log2(2 * width)))
    xc = torch.fft.irfft(torch.conj(torch.fft.rfft(a, nfft)) * torch.fft.rfft(b, nfft), nfft)
    xc = torch.cat([xc[..., -MAXLAG:], xc[..., :MAXLAG + 1]], -1)
    return xc / (torch.sqrt((a * a).sum(-1) * (b * b).sum(-1)).unsqueeze(-1) + 1e-30)


def _plane_tiles(q):
    from ml4gw.transforms.qtransform import QTile, SingleQTransform
    fmin = max(20.0, q / (2 * math.pi * 0.25))
    fmax = min(1950.0, SR / 2 / (1 + math.sqrt(11) / q))
    ns = types.SimpleNamespace(frange=[fmin, fmax], q=q, mismatch=0.2, duration=1.0)
    return [QTile(q=q, frequency=float(f), duration=1.0, sample_rate=SR, mismatch=0.2)
            for f in SingleQTransform.get_freqs(ns)]


class _Plane(nn.Module):
    """Gather/window tables of one Q plane; rows with the same number of tiles are transformed together."""

    def __init__(self, q, line_hz, alpha):
        super().__init__()
        self.q = q
        tiles = _plane_tiles(q)
        R = len(tiles)
        freq = np.array([t.frequency for t in tiles])
        n_row = np.array([t.ntiles() for t in tiles])
        self.group_shapes = []                  # (rows in group, tiles per row)
        order = []
        for g, n in enumerate(sorted(set(n_row.tolist()))):
            rows = np.flatnonzero(n_row == n)
            idx = np.full((len(rows), n), N // 2 + 1, np.int64)        # N/2 + 1 = appended zero bin
            win = np.zeros((len(rows), n), np.float32)
            for j, r in enumerate(rows):
                t = tiles[r]
                left = int(t.padding[0])
                idx[j, left:left + t.windowsize] = t.indices.numpy()
                win[j, left:left + t.windowsize] = t.window.numpy()
            # QTile.forward applies ifftshift to the padded vector before the ifft
            self.register_buffer(f"idx{g}", torch.as_tensor(np.fft.ifftshift(idx, axes=-1)))
            self.register_buffer(f"win{g}", torch.as_tensor(np.fft.ifftshift(win, axes=-1)))
            self.group_shapes.append((len(rows), int(n)))
            order += rows.tolist()
        order = np.array(order)
        off = np.zeros(R, np.int64)
        off[order] = np.concatenate([[0], np.cumsum(n_row[order])[:-1]])
        self.ntot = int(n_row.sum())
        self.K = min(N_MAX, self.ntot)
        self.register_buffer("first", torch.arange(self.K) < N_MIN)
        self.register_buffer("unsort", torch.as_tensor(np.argsort(order)))      # group order -> row order
        self.register_buffer("row_of", torch.as_tensor(np.repeat(order, n_row[order])))
        self.register_buffer("t_of", torch.as_tensor(np.concatenate([np.arange(n_row[r]) for r in order])))
        self.register_buffer("off", torch.as_tensor(off))
        self.register_buffer("n_row", torch.as_tensor(n_row))
        self.register_buffer("freq", torch.as_tensor(freq, dtype=torch.float32))
        # carrier frequency of each row (centre bin of its window, Hz for 1 s kernels)
        fdem = np.array([float(t.indices[t.windowsize // 2]) for t in tiles])
        self.register_buffer("fdem", torch.as_tensor(fdem, dtype=torch.float32))
        # tile weight n_row * df_row / sum(window^2) * alpha
        S = np.array([float((t.window.double() ** 2).sum()) for t in tiles])
        df = np.gradient(freq) if R > 1 else np.array([freq[0]])
        w0 = torch.as_tensor(n_row * df / S, dtype=torch.float32) * alpha
        self.register_buffer("w", w0)
        line = np.zeros(R, bool)
        for f0 in line_hz:
            line[int(np.abs(freq - f0).argmin())] = True
        self.register_buffer("line", torch.as_tensor(line))
        # nearest-tile shift of each lag on each row: round(tau * n_row / N), half-to-even
        self.register_buffer("shift", torch.as_tensor(np.round(LAGS[None] * n_row[:, None] / N).astype(np.int64)))
        ph = -2 * np.pi * fdem[:, None] * LAGS[None] / N
        self.register_buffer("ph_re", torch.as_tensor(np.cos(ph), dtype=torch.float32))
        self.register_buffer("ph_im", torch.as_tensor(np.sin(ph), dtype=torch.float32))

    def forward(self, X):
        """X: (B, 2, N/2 + 2) complex spectrum -> dict of (B,) plane features"""
        B = X.shape[0]
        W, rm = [], []
        for g, (rg, n) in enumerate(self.group_shapes):
            V = X[..., getattr(self, f"idx{g}")] * getattr(self, f"win{g}")          # (B, 2, rg, n)
            Wg = torch.fft.ifft(V, dim=-1)
            rm.append((Wg.real ** 2 + Wg.imag ** 2).median(-1).values)              # per-row median power
            W.append(Wg.reshape(B, 2, rg * n))
        W = torch.cat(W, -1)                                                         # (B, 2, ntot)
        rm = torch.cat(rm, -1)[..., self.unsort]                                     # (B, 2, R)
        P = W.real ** 2 + W.imag ** 2
        dead = rm < DEAD * rm.median(-1, keepdim=True).values
        live = ~dead.any(1)                                                          # (B, R)
        glob = torch.nanmedian(torch.where(live[:, None], rm, torch.full_like(rm, float("nan"))), -1, keepdim=True).values
        glob = torch.nan_to_num(glob, nan=0.0)                                       # no live row (flat input)
        scale = torch.where(self.line[None, None], rm, glob.expand_as(rm)) / math.log(2) + 1e-12
        sc_t = scale[..., self.row_of]
        lv_t = live[:, self.row_of].float()
        e = P / sc_t * lv_t[:, None]
        Wn = W / sc_t.sqrt() * lv_t[:, None]
        S = e[:, 0] + e[:, 1]
        vals, idx = torch.topk(S, self.K, dim=-1)                                    # loudest network tiles
        m = ((vals >= E_THR) | self.first[None]).float()                          # >= E_THR, at least N_MIN tiles
        row = self.row_of[idx]
        w = self.w[row] * m
        eH = torch.gather(e[:, 0], 1, idx)
        WH = torch.gather(Wn[:, 0], 1, idx)
        n = self.n_row[row]
        j = self.off[row].unsqueeze(-1) + torch.remainder(self.t_of[idx].unsqueeze(-1) + self.shift[row], n.unsqueeze(-1))
        WL = torch.gather(Wn[:, 1], 1, j.reshape(B, -1)).reshape(j.shape)            # (B, K, L)
        eL = torch.gather(e[:, 1], 1, j.reshape(B, -1)).reshape(j.shape)
        ph = torch.complex(self.ph_re[row], self.ph_im[row])
        C = (w.unsqueeze(-1) * WH.unsqueeze(-1) * WL.conj() * ph).sum(1)            # (B, L)
        EL = (w.unsqueeze(-1) * eL).sum(1)
        EH = (w * eH).sum(1)
        aC = C.abs()
        cce = 2 * aC / (EH.unsqueeze(-1) + EL + 1e-12)
        i = cce.argmax(-1, keepdim=True)
        aCi = torch.gather(aC, 1, i)[:, 0]
        ELi = torch.gather(EL, 1, i)[:, 0]
        EL0 = EL[:, MAXLAG]
        kh = e[:, 0].argmax(-1)
        kl = e[:, 1].argmax(-1)
        th = self.t_of[kh].float() / self.n_row[self.row_of[kh]].float()
        tl = self.t_of[kl].float() / self.n_row[self.row_of[kl]].float()
        return {
            "ccen_tf": torch.gather(cce, 1, i)[:, 0],
            "wcc2_tf": aCi / torch.sqrt(EH * ELi + 1e-24),
            "ecoh_tf": aCi,
            "e_clu_H": EH,
            "e_clu_L": EL0,
            "balance": torch.abs(torch.log((EH + 1e-6) / (EL0 + 1e-6))),
            "enet_peak": vals[:, 0],
            "epeak_H": torch.gather(e[:, 0], 1, kh.unsqueeze(-1))[:, 0],
            "epeak_L": torch.gather(e[:, 1], 1, kl.unsqueeze(-1))[:, 0],
            "dt_peak_ms": torch.abs(th - tl) * 1e3,
            "dlogf_peak": torch.abs(torch.log(self.freq[self.row_of[kh]] / self.freq[self.row_of[kl]])),
        }


class CorrelationFeatures(nn.Module):
    """(B, 2, 4096) whitened H1/L1 kernels -> (B, F) correlation features, F = len(FEATURES[mode])."""

    def __init__(self, mode="corr21", line_hz=None, alphas=None):
        super().__init__()
        if mode not in FEATURES:
            raise ValueError(f"mode must be one of {sorted(FEATURES)}, got {mode!r}")
        self.mode = mode
        self.features = list(FEATURES[mode])
        line_hz = LINE_HZ if line_hz is None else line_hz
        alphas = ALPHAS if alphas is None else alphas
        self.planes = nn.ModuleList([_Plane(q, line_hz.get(q, ()), alphas[k]) for k, q in enumerate(QS)])
        self.register_buffer("band_gain", torch.as_tensor(np.stack([_band_gain(lo, hi) for lo, hi in BANDS]), dtype=torch.float32))
        self.register_buffer("band_z", torch.as_tensor([math.sqrt(2 * (hi - lo)) for lo, hi in BANDS], dtype=torch.float32))
        self.register_buffer("env_gain", torch.as_tensor(_band_gain(*ENV_BAND), dtype=torch.float32))
        hil = np.zeros(N)                                           # analytic-signal multiplier on the full FFT
        hil[0] = hil[N // 2] = 1.0
        hil[1:N // 2] = 2.0
        self.register_buffer("hilbert", torch.as_tensor(hil, dtype=torch.float32))
        self.register_buffer("hann", torch.as_tensor(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN), dtype=torch.float32))
        fseg = np.fft.rfftfreq(WIN, 1 / SR)
        self.register_buffer("msc_bins", torch.as_tensor(np.flatnonzero((fseg >= MSC_FRANGE[0]) & (fseg <= MSC_FRANGE[1]))))
        self.register_buffer("lags_ms", torch.as_tensor(LAGS / SR * 1e3, dtype=torch.float32))
        self.register_buffer("tf_scale", torch.as_tensor(np.r_[1.0, np.full(N // 2, 2.0)], dtype=torch.float32))
        for w in ENV_HALF:
            self.register_buffer(f"env_idx{w}", torch.arange(2 * w))
        self.register_buffer("lag_idx", torch.as_tensor(LAGS))
        self.register_buffer("boxcar", torch.full((1, 1, ENV_SMOOTH), 1.0 / ENV_SMOOTH))

    @property
    def num_features(self):
        return len(self.features)

    def time_domain(self, x):
        """x: (B, 2, N) standardised float32"""
        B = x.shape[0]
        H, L = x[:, 0], x[:, 1]
        out = {}
        rho = _lagged_ncc(H, L, N)
        out["pearson_0lag"] = rho[:, MAXLAG].abs()
        out["cc_maxlag_1s"] = rho.abs().amax(1)
        out["cc_maxlag_win512"] = _lagged_ncc(H.unfold(-1, WIN, HOP), L.unfold(-1, WIN, HOP), WIN).abs().flatten(1).amax(1)
        Xf = torch.fft.rfft(x)                                                     # (B, 2, N/2 + 1)
        xb = torch.fft.irfft(Xf.unsqueeze(1) * self.band_gain[None, :, None], N)   # (B, 6, 2, N)
        out["cc_band_maxz"] = (_lagged_ncc(xb[:, :, 0], xb[:, :, 1], N).abs().amax(-1) * self.band_z).amax(1)
        # Welch magnitude-squared coherence (scipy.signal.coherence, hann, nperseg 512, noverlap 256)
        seg = x.unfold(-1, WIN, HOP)
        seg = (seg - seg.mean(-1, keepdim=True)) * self.hann
        Fs = torch.fft.rfft(seg)                                                   # (B, 2, nseg, WIN/2 + 1)
        Pxx = (Fs.real ** 2 + Fs.imag ** 2).mean(2)
        Pxy = (Fs[:, 0].conj() * Fs[:, 1]).mean(1)
        msc = (Pxy.real ** 2 + Pxy.imag ** 2) / (Pxx[:, 0] * Pxx[:, 1] + 1e-30)
        out["msc_max"] = msc[:, self.msc_bins].amax(1)
        # one window around the loudest smoothed H+L Hilbert envelope point, max-lag |NCC| with L truly shifted
        Xe = Xf * self.env_gain
        xe = torch.fft.irfft(Xe, N)                                                # (B, 2, N)
        Z = torch.fft.ifft(torch.cat([Xe, torch.zeros_like(Xe[..., 1:N // 2])], -1) * self.hilbert)
        env = (Z.real ** 2 + Z.imag ** 2).sum(1, keepdim=True)                     # (B, 1, N)
        sm = nn.functional.conv1d(env, self.boxcar, padding=ENV_SMOOTH // 2)[:, 0]
        pad = max(ENV_HALF) + MAXLAG
        xl = nn.functional.pad(xe[:, 1], (pad, pad))
        for w in ENV_HALF:
            c = sm[:, w:N - w].argmax(1) + w                                       # window centre
            ar = getattr(self, f"env_idx{w}")
            a = torch.gather(xe[:, 0], 1, (c - w).unsqueeze(1) + ar[None])          # (B, 2w)
            a = a - a.mean(-1, keepdim=True)
            na = torch.sqrt((a * a).sum(-1))
            j = (pad + c - w)[:, None, None] + self.lag_idx[None, :, None] + ar[None, None]   # (B, L, 2w)
            v = torch.gather(xl, 1, j.reshape(B, -1)).reshape(j.shape)
            v = v - v.mean(-1, keepdim=True)
            r = (a.unsqueeze(1) * v).sum(-1) / (na.unsqueeze(1) * torch.sqrt((v * v).sum(-1)) + 1e-30)
            k = r.abs().argmax(-1, keepdim=True)
            out[f"xc_env_w{w}"] = torch.gather(r.abs(), 1, k)[:, 0]
            if w == 256:
                out["xc_env_w256_lag_ms"] = self.lags_ms[k[:, 0]].abs()
        return out

    def time_frequency(self, x):
        """x: (B, 2, N) standardised float32"""
        X = torch.fft.rfft(x, norm="forward") * self.tf_scale                      # ml4gw/GWpy scaling
        X = torch.cat([X, torch.zeros_like(X[..., :1])], -1)                       # zero bin for padding gathers
        res = [pl(X) for pl in self.planes]
        best = torch.stack([r["enet_peak"] for r in res]).argmax(0, keepdim=True)  # plane of the loudest tile
        keys = [k for k in TF_COH_FEATURES + TF_LOUD_FEATURES if k != "ccen_tf_reg"]
        out = {k: torch.gather(torch.stack([r[k] for r in res]), 0, best)[0] for k in keys}
        etot = 2 * out["ecoh_tf"] / torch.clamp(out["ccen_tf"], min=1e-6)
        out["ccen_tf_reg"] = 2 * out["ecoh_tf"] / (etot + 20.0)
        return out

    def forward(self, x):
        # exact power-of-two prescale per event/detector (bit-identical to plain float32 at O(1) scale; strain-scale
        # input would otherwise underflow in the float32 variance)
        x = x.to(torch.float64)
        amax = x.abs().amax(-1, keepdim=True)
        e = torch.where(amax > 0, -torch.floor(torch.log2(amax)), torch.zeros_like(amax))
        x = _standardise(torch.ldexp(x, e).to(torch.float32))
        out = self.time_domain(x)
        out.update(self.time_frequency(x))
        return torch.stack([out[k] for k in self.features], 1)


_CACHE = {}


def correlation_features(batch, mode="corr21"):
    """Functional form for transforms.frequency_cos_similarity: one cached CorrelationFeatures per (mode, device)."""
    key = (mode, str(batch.device))
    if key not in _CACHE:
        _CACHE[key] = CorrelationFeatures(mode).to(batch.device).eval()
    with torch.no_grad():
        return _CACHE[key](batch)
