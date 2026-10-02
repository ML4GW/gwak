"""matching.py — cross-detector tile matching, time-of-flight, best-lag (DESIGN §4).

Shared-grid exact cell match: H and L live on the SAME (plane, q, f0) lattice, so a cell
matches by exact (row_idx) equality with |t_H - (t_L + tau)| <= tau_tof.

The lag tau enters ONLY as a per-tile phase rotation z_L'(c) = z_L(c) * exp(i 2 pi f0_c tau)
(Regime A), applied identically in S_HL and the best-lag scan.

best-lag — TWO STAGES (ruling Q1, DESIGN §4):
  1. ALIGN (carrier-free): best_lag = argmax_tau |S_HL(tau)|  (the magnitude/envelope of the
     coherent sum discards the carrier phase and peaks at the true envelope alignment, NOT a
     carrier sidelobe). Regime B analogue: energy cross-power sum_Omega sqrt(E_H E_L(c+tau)).
  2. READ OUT (signed, at the fixed lag): S_HL / cc / R / xcorr are evaluated at the one fixed
     best_lag via the per-tile phase rotation -- the rotation reads coherence, never finds it.
  bestlag_unconstrained=1 when (a participating tile's duration > scan range) OR (envelope
  contrast < contrast_min); then coherence is read at best_lag = 0 (zero-lag), an unbiased
  defined alignment.

The MatchedSet carries aligned arrays (xH, xL', EH, EL) over the matched region Omega plus the
matched clusters used by the occupancy gate (ruling Q2).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tiles import TileSet


@dataclass
class MatchedCluster:
    cell_idx: np.ndarray          # i8 indices into the MatchedSet aligned arrays
    t_center: float
    f0_center: float
    q_center: float
    E_H: float
    E_L: float
    n_tiles: int


@dataclass
class MatchedSet:
    omega_row_ids: np.ndarray     # i8 grid row index per matched cell
    omega_f0: np.ndarray          # f8 f0 per matched cell
    omega_t: np.ndarray           # f8 H-tile center time per matched cell
    xH: Optional[np.ndarray]      # c16 aligned H amps over Omega (Regime A) | None
    xL: Optional[np.ndarray]      # c16 aligned L amps, already tau-rephased | None
    EH: np.ndarray                # f8 |z_H|^2 over Omega
    EL: np.ndarray                # f8 |z_L'|^2 over Omega
    n_matched: int
    lag: float
    regime: str
    clusters: List[MatchedCluster] = field(default_factory=list)
    dt_match: Optional[np.ndarray] = None   # f8 |t_H - (t_L + lag)| per matched cell

    @property
    def empty(self) -> bool:
        return self.n_matched == 0


def _index_by_row(ts: TileSet) -> Dict[int, List]:
    """Map row_idx -> list of (time, time_bin, energy, complex_amp, f0, q)."""
    idx: Dict[int, List] = {}
    for t in ts.tiles:
        idx.setdefault(t.row_idx, []).append(
            (t.t, t.time_bin, t.energy, t.complex_amp, t.f0, t.q))
    return idx


def _cluster_matched(row_ids, ts, ts_q, t_arr, f0_arr, EH, EL, config) -> List[MatchedCluster]:
    """Group matched cells into clusters within dt_cluster in time (and same grid row).

    cWB pixel-cluster / oLIB coincident-cluster object used by the occupancy gate (Q2).
    """
    n = len(t_arr)
    if n == 0:
        return []
    # group by (row_idx); within a row, cells contiguous in time within dt_cluster
    order = np.argsort(t_arr, kind="stable")
    clusters: List[MatchedCluster] = []
    # bucket by row first
    by_row: Dict[int, List[int]] = {}
    for i in order:
        by_row.setdefault(int(row_ids[i]), []).append(int(i))
    # but clusters in DESIGN are time-contiguous (and contiguous freq); we cluster across rows
    # by time proximity, keeping it simple and faithful: sort all by time, split on gaps > dt.
    dt = config.dt_cluster
    cur: List[int] = []
    last_t = None
    for i in order:
        ti = t_arr[i]
        if last_t is None or (ti - last_t) <= dt:
            cur.append(int(i))
        else:
            clusters.append(_make_cluster(cur, t_arr, f0_arr, ts_q, EH, EL))
            cur = [int(i)]
        last_t = ti
    if cur:
        clusters.append(_make_cluster(cur, t_arr, f0_arr, ts_q, EH, EL))
    return clusters


def _make_cluster(idxs, t_arr, f0_arr, q_arr, EH, EL) -> MatchedCluster:
    idxs = np.asarray(idxs, dtype=np.int64)
    return MatchedCluster(
        cell_idx=idxs,
        t_center=float(np.mean(t_arr[idxs])),
        f0_center=float(np.mean(f0_arr[idxs])),
        q_center=float(np.mean(q_arr[idxs])),
        E_H=float(np.sum(EH[idxs])),
        E_L=float(np.sum(EL[idxs])),
        n_tiles=int(idxs.size),
    )


def match_tiles(ts_H: TileSet, ts_L: TileSet, config, lag: float = 0.0) -> MatchedSet:
    """Form the matched region Omega on the shared grid at a given lag.

    For each shared (row_idx), pair H tiles with L tiles whose time satisfies
    |t_H - (t_L + lag)| <= tau_tof. Each pair contributes one matched cell; L is
    tau-rephased by exp(i 2 pi f0 lag) (Regime A).
    """
    regime_A = (config.regime == "A")
    regime_label = "A_complex_faithful" if regime_A else "B_unsigned_degraded"

    idx_H = _index_by_row(ts_H)
    idx_L = _index_by_row(ts_L)
    tof = config.tau_tof

    row_ids: List[int] = []
    f0s: List[float] = []
    qs: List[float] = []
    tHs: List[float] = []
    dts: List[float] = []
    xH_l: List[complex] = []
    xL_l: List[complex] = []
    EH_l: List[float] = []
    EL_l: List[float] = []

    for row_idx, hlist in idx_H.items():
        if row_idx not in idx_L:
            continue
        llist = idx_L[row_idx]
        llist_sorted = sorted(llist, key=lambda r: r[0])
        ltimes = np.array([r[0] for r in llist_sorted])
        for (tH, tbH, eH, cH, f0H, qH) in hlist:
            lo = tH - lag - tof
            hi = tH - lag + tof
            sel = np.where((ltimes >= lo) & (ltimes <= hi))[0]
            if sel.size == 0:
                continue
            best = sel[np.argmin(np.abs(ltimes[sel] - (tH - lag)))]
            (tL, tbL, eL, cL, f0L, qL) = llist_sorted[best]
            row_ids.append(row_idx)
            f0s.append(f0H)
            qs.append(qH)
            tHs.append(tH)
            dts.append(abs(tH - (tL + lag)))
            EH_l.append(eH)
            EL_l.append(eL)
            if regime_A and cH is not None and cL is not None:
                cLp = cL * np.exp(1j * 2.0 * np.pi * f0H * lag)
                xH_l.append(cH)
                xL_l.append(cLp)

    n = len(row_ids)
    if regime_A and n > 0 and len(xH_l) == n:
        xH = np.asarray(xH_l, dtype=np.complex128)
        xL = np.asarray(xL_l, dtype=np.complex128)
    else:
        xH = None
        xL = None

    EH = np.asarray(EH_l, dtype=np.float64)
    EL = np.asarray(EL_l, dtype=np.float64)
    row_ids_a = np.asarray(row_ids, dtype=np.int64)
    f0_a = np.asarray(f0s, dtype=np.float64)
    q_a = np.asarray(qs, dtype=np.float64)
    t_a = np.asarray(tHs, dtype=np.float64)
    dt_a = np.asarray(dts, dtype=np.float64)

    clusters = _cluster_matched(row_ids_a, ts_H, q_a, t_a, f0_a, EH, EL, config) if n else []

    return MatchedSet(
        omega_row_ids=row_ids_a,
        omega_f0=f0_a,
        omega_t=t_a,
        xH=xH, xL=xL,
        EH=EH, EL=EL,
        n_matched=n, lag=lag, regime=regime_label,
        clusters=clusters, dt_match=dt_a,
    )


def s_hl(matched: MatchedSet, config) -> float:
    """The shared cross-term primitive S_HL = sum_Omega Re[z_H conj(z_L')] (Regime A),
    or sum_Omega sqrt(E_H E_L) (Regime B, unsigned proxy)."""
    if matched.empty:
        return np.nan
    if config.regime == "A":
        if matched.xH is None or matched.xL is None:
            return np.nan
        return float(np.sum(np.real(matched.xH * np.conj(matched.xL))))
    else:
        return float(np.sum(np.sqrt(matched.EH * matched.EL)))


def _envelope(matched: MatchedSet, config) -> float:
    """Carrier-free envelope A(tau) = |sum_Omega z_H conj(z_L')| (Regime A), or the energy
    cross-power sum_Omega sqrt(E_H E_L) (Regime B)."""
    if matched.empty:
        return np.nan
    if config.regime == "A":
        if matched.xH is None or matched.xL is None:
            return np.nan
        return float(np.abs(np.sum(matched.xH * np.conj(matched.xL))))
    else:
        return float(np.sum(np.sqrt(matched.EH * matched.EL)))


@dataclass
class LagResult:
    best_lag: float
    xcorr_bestlag: float
    xcorr_zerolag: float
    s_hl_bestlag: float
    curve_tau: np.ndarray
    curve_C: np.ndarray            # signed normalized xcorr per tau (diagnostic)
    curve_env: np.ndarray          # carrier-free envelope per tau (the align curve)
    unconstrained: bool


def _normalized_xcorr(matched: MatchedSet, config) -> float:
    """Signed normalized cross-power S_HL / sqrt(E_H E_L) over Omega (read-out at fixed lag)."""
    if matched.empty:
        return np.nan
    EH = float(np.sum(matched.EH))
    EL = float(np.sum(matched.EL))
    denom = np.sqrt(EH * EL)
    if denom <= 0:
        return np.nan
    return s_hl(matched, config) / denom


def best_lag_scan(ts_H: TileSet, ts_L: TileSet, config) -> LagResult:
    """Two-stage best-lag (ruling Q1).

    STAGE 1 (align, carrier-free): best_lag = argmax_tau |S_HL(tau)| (the envelope).
    STAGE 2 (read out): the signed normalized xcorr C is reported at best_lag and at zero-lag.

    bestlag_unconstrained=1 when a participating tile's duration > scan range OR the envelope
    contrast (max-median)/max < contrast_min; then best_lag is forced to 0 (zero-lag read-out).
    """
    taus = np.arange(-config.lag_window, config.lag_window + 0.5 * config.lag_step,
                     config.lag_step)
    env_vals = np.full(taus.size, np.nan)
    c_vals = np.full(taus.size, np.nan)
    s_vals = np.full(taus.size, np.nan)
    for k, tau in enumerate(taus):
        m = match_tiles(ts_H, ts_L, config, lag=float(tau))
        env_vals[k] = _envelope(m, config)
        c_vals[k] = _normalized_xcorr(m, config)
        s_vals[k] = s_hl(m, config)

    kzero = int(np.argmin(np.abs(taus)))

    if np.all(np.isnan(env_vals)):
        return LagResult(0.0, np.nan, np.nan, np.nan, taus, c_vals, env_vals, False)

    # STAGE 1: carrier-free envelope argmax
    kbest = int(np.nanargmax(env_vals))
    best_lag = float(taus[kbest])

    # bestlag_unconstrained:
    # (i) envelope contrast below threshold (narrowband -> lag unresolvable)
    finite_env = env_vals[np.isfinite(env_vals)]
    contrast = np.nan
    if finite_env.size and np.nanmax(env_vals) > 0:
        emax = float(np.nanmax(env_vals))
        emed = float(np.nanmedian(finite_env))
        contrast = (emax - emed) / emax
    low_contrast = bool(np.isfinite(contrast) and contrast < config.contrast_min)

    # (ii) the cluster's CHARACTERISTIC (median participating-tile) duration exceeds the scan
    #      range -> the cluster is long-duration/narrowband, lag unresolvable. We use the median
    #      tile duration (the characteristic cluster timescale), NOT "any single tile": a genuine
    #      broadband burst always contains some low-f0/high-Q long tiles, so an "any-tile" test
    #      would flag every broadband event (which the contrast test correctly does NOT). The
    #      median cleanly separates narrowband (>10 ms) from broadband (<10 ms), matching the
    #      ruling's "long-duration tile vs scan range" intent and the envelope-contrast result.
    m_best = match_tiles(ts_H, ts_L, config, lag=best_lag)
    dur_exceeds = False
    if not m_best.empty:
        durs = np.array([ts_H.grid.rows[r].duration for r in m_best.omega_row_ids])
        dur_exceeds = bool(np.median(durs) > config.lag_window)

    unconstrained = low_contrast or dur_exceeds
    if unconstrained:
        # read coherence at zero-lag, an unbiased defined alignment
        best_lag = 0.0
        kbest = kzero

    return LagResult(
        best_lag=best_lag,
        xcorr_bestlag=float(c_vals[kbest]),
        xcorr_zerolag=float(c_vals[kzero]),
        s_hl_bestlag=float(s_vals[kbest]),
        curve_tau=taus, curve_C=c_vals, curve_env=env_vals,
        unconstrained=unconstrained,
    )
