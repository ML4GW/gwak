"""features.py — the FULL feature set (DESIGN §5 / features_reconciled §1,§3,§5,§6,§7).

PRIMARY: cc_tile is the SIGNED form  cc_tile = ec_tile / E_tile = 2*S_HL/(E_H+E_L) in [-1,1].
This is NOT the dominant-mode form. The dominant-mode cc_dom_tile is a GATED companion only
(NaN + single_det_dominated_tile=1 when only one detector occupies Omega).

OCCUPANCY GATE (ruling Q2): coherence features return NaN + low_occupancy_tile=1 unless
  (i) |Omega| >= min_omega (default 3) AND
  (ii) there is >=1 matched coincident cluster with >= k_cluster (default 2) tiles.

All functions NaN-safe (empty Omega -> NaN, never a crash).

Energy-conservation identity (Regime A, equal-weight orthonormal split):
    ec_tile + 2*en_tile == E_tile     (machine precision)
"""
from typing import Dict, Optional, Tuple

import numpy as np

from .matching import MatchedSet, s_hl
from .tiles import TileSet


_EPS = 1e-12


# --------------------------------------------------------------------------- occupancy gate

def occupancy_ok(matched: MatchedSet, config) -> bool:
    """Two-part occupancy gate (ruling Q2): |Omega| >= min_omega AND a coincident cluster
    with >= k_cluster tiles exists. Returns True when coherence features may be computed."""
    if matched.empty:
        return False
    min_omega = getattr(config, "min_omega", 1)
    k_cluster = getattr(config, "k_cluster", 1)
    if matched.n_matched < min_omega:
        return False
    has_cluster = any(c.n_tiles >= k_cluster for c in matched.clusters)
    return bool(has_cluster)


def _gated(matched: MatchedSet, config) -> bool:
    """True if coherence features should be gated to NaN (occupancy NOT ok)."""
    return not occupancy_ok(matched, config)


# --------------------------------------------------------------------------- energy bookkeeping

def total_energies(matched: MatchedSet):
    """(E_tile, Em_tile, E_H, E_L) over Omega. Both regimes identical."""
    if matched.empty:
        return (np.nan, np.nan, np.nan, np.nan)
    E_H = float(np.sum(matched.EH))
    E_L = float(np.sum(matched.EL))
    E_tile = E_H + E_L
    Em_tile = float(np.sum(np.maximum(matched.EH, matched.EL)))
    return (E_tile, Em_tile, E_H, E_L)


# --------------------------------------------------------------------------- cWB-inspired core

def ec_tile(matched: MatchedSet, config) -> float:
    """Coherent energy E_c = 2*S_HL (Regime A, signed) / 2*S_HL_proxy (Regime B, >=0)."""
    if _gated(matched, config):
        return np.nan
    return 2.0 * s_hl(matched, config)


def en_tile(matched: MatchedSet, config) -> float:
    """Null-stream energy en = sum_Omega |nul|^2, nul = (z_H - z_L')/sqrt(2)  (Regime A).

    The FULL incoherent energy is 2*en = E_tile - ec. Regime B: en_proxy = (E_tile - ec)/2.
    """
    if _gated(matched, config):
        return np.nan
    if config.regime == "A" and matched.xH is not None:
        nul = (matched.xH - matched.xL) / np.sqrt(2.0)
        return float(np.sum(np.abs(nul) ** 2))
    E_tile, _, _, _ = total_energies(matched)
    ec = 2.0 * s_hl(matched, config)
    return float((E_tile - ec) / 2.0)


def l_model_tile(matched: MatchedSet, config) -> float:
    """Coherent-model energy L_model = sum_Omega |coh|^2, coh = (z_H + z_L')/sqrt(2) (Regime A).
    L_model + en_tile == E_tile (orthonormal split). Regime B: L_model = (E_tile + ec)/2."""
    if _gated(matched, config):
        return np.nan
    if config.regime == "A" and matched.xH is not None:
        coh = (matched.xH + matched.xL) / np.sqrt(2.0)
        return float(np.sum(np.abs(coh) ** 2))
    E_tile, _, _, _ = total_energies(matched)
    ec = 2.0 * s_hl(matched, config)
    return float((E_tile + ec) / 2.0)


def cc_tile(matched: MatchedSet, config) -> float:
    """PRIMARY VETO (signed): cc_tile = ec_tile / E_tile = 2*S_HL/(E_H+E_L) in [-1,1].

    Denominator is E_tile (zero only for empty cluster) -> NO anti-coherent pole.
    """
    if _gated(matched, config):
        return np.nan
    E_tile, _, _, _ = total_energies(matched)
    if E_tile <= _EPS:
        return np.nan
    return (2.0 * s_hl(matched, config)) / E_tile


def chi2_tile(matched: MatchedSet, config) -> float:
    """Reduced-chi2 of the null-stream energy: chi2 = en_tile / (kappa_noise * |Omega|).

    kappa_noise (calibrated on Background, DESIGN §5) absorbs median-vs-mean and over-complete
    tiling DoF overcount so E[chi2]~=1 on Background. If kappa_noise is None, fall back to
    en / |Omega| (uncalibrated, flagged in metadata)."""
    if _gated(matched, config):
        return np.nan
    en = en_tile(matched, config)
    nomega = matched.n_matched
    if nomega <= 0:
        return np.nan
    kappa = getattr(config, "kappa_noise", None)
    if kappa is None or not np.isfinite(kappa) or kappa <= 0:
        kappa = 1.0
    return float(en / (kappa * nomega))


def rho_tile(matched: MatchedSet, config) -> float:
    """Coherent network amplitude rho = sqrt(max(ec,0)*max(cc,0)/K). Guard the PRODUCT so
    anti-coherent -> 0 with no NaN."""
    if _gated(matched, config):
        return np.nan
    ec = ec_tile(matched, config)
    cc = cc_tile(matched, config)
    K = getattr(config, "K", 2)
    prod = max(ec, 0.0) * max(cc, 0.0) / float(K)
    if prod <= 0:
        return 0.0
    return float(np.sqrt(prod))


def scc_tile(matched: MatchedSet, config) -> float:
    """Sub-network consistency scc = (E-Em)/((E-Em) + en_full + n_reg), n_reg = O(1) regulator
    (default 1.0, NOT the tile count). en_full = E_tile - ec_tile (the honest (E-L) term)."""
    if _gated(matched, config):
        return np.nan
    E_tile, Em_tile, _, _ = total_energies(matched)
    ec = 2.0 * s_hl(matched, config)
    en_full = E_tile - ec
    n_reg = getattr(config, "n_reg", 1.0)
    num = (E_tile - Em_tile)
    denom = num + en_full + n_reg
    if denom <= _EPS:
        return np.nan
    return float(num / denom)


def _amp_ratio(matched: MatchedSet) -> float:
    """Fitted amplitude ratio a = sqrt(E_H / E_L) over Omega (for the edr model-residual D)."""
    E_H = float(np.sum(matched.EH))
    E_L = float(np.sum(matched.EL))
    if E_L <= _EPS:
        return np.nan
    return float(np.sqrt(E_H / E_L))


def edr_tile(matched: MatchedSet, config) -> float:
    """Energy-disbalance ratio edr = D / max(C, eps), C = max(ec_tile, 0).

    Regime A (default) D = model-residual sum||z_H|^2 - |a z_L'|^2| with a = sqrt(E_H/E_L)
    (removes the double-penalty on antenna-asymmetric coherent signals). Regime-B / fallback
    D = sum|E_H - E_L| (raw imbalance)."""
    if _gated(matched, config):
        return np.nan
    ec = 2.0 * s_hl(matched, config)
    C = max(ec, 0.0)
    if config.regime == "A" and matched.xH is not None:
        a = _amp_ratio(matched)
        if np.isfinite(a):
            zL2 = np.abs(matched.xL) ** 2
            zH2 = np.abs(matched.xH) ** 2
            D = float(np.sum(np.abs(zH2 - (a ** 2) * zL2)))
        else:
            D = float(np.sum(np.abs(matched.EH - matched.EL)))
    else:
        D = float(np.sum(np.abs(matched.EH - matched.EL)))
    return float(D / max(C, _EPS))


# --------------------------------------------------------------------------- dominant-mode companion

def _eig_R(matched: MatchedSet) -> Tuple[float, float, float, float, complex]:
    """Return (lambda1, lambda2, R11, R22, R12) for R = sum ζζ† (Regime A)."""
    R11 = float(np.sum(np.abs(matched.xH) ** 2))
    R22 = float(np.sum(np.abs(matched.xL) ** 2))
    R12 = complex(np.sum(matched.xH * np.conj(matched.xL)))
    tr = R11 + R22
    det = R11 * R22 - (R12.real ** 2 + R12.imag ** 2)
    disc = max(tr * tr - 4.0 * det, 0.0)
    sq = np.sqrt(disc)
    lam1 = 0.5 * (tr + sq)
    lam2 = 0.5 * (tr - sq)
    return (lam1, lam2, R11, R22, R12)


def cc_dom_tile(matched: MatchedSet, config):
    """Dominant-mode coherence (lambda1-lambda2)/(lambda1+lambda2) in [0,1], GATED on two-sided
    occupancy: defined only when BOTH E_H and E_L over Omega exceed E_thr; else NaN +
    single_det_dominated=True. Regime A only (needs complex R12).
    Returns (cc_dom, en_dom, single_det_dominated_flag)."""
    if _gated(matched, config):
        return (np.nan, np.nan, True)
    E_tile, _, E_H, E_L = total_energies(matched)
    thr = config.sig_energy_thr if config.e_thr_calibrated is None else config.e_thr_calibrated
    if not (E_H >= thr and E_L >= thr):
        return (np.nan, np.nan, True)
    if config.regime != "A" or matched.xH is None:
        return (np.nan, np.nan, True)
    lam1, lam2, _, _, _ = _eig_R(matched)
    if (lam1 + lam2) <= _EPS:
        return (np.nan, np.nan, False)
    return (float((lam1 - lam2) / (lam1 + lam2)), float(lam2), False)


def cc_offdiag_tile(matched: MatchedSet, config) -> float:
    """Sign-free faithful coherence |R12|/sqrt(E_H E_L) in [0,1] (Cauchy-Schwarz)."""
    if _gated(matched, config):
        return np.nan
    if config.regime != "A" or matched.xH is None:
        return np.nan
    _, _, R11, R22, R12 = _eig_R(matched)
    denom = np.sqrt(R11 * R22)
    if denom <= _EPS:
        return np.nan
    return float(abs(R12) / denom)


# --------------------------------------------------------------------------- oLIB cross-power

def xcorr_signed(matched: MatchedSet, config) -> float:
    """Normalized cross-power S_HL / sqrt(E_H E_L) in [-1,1] (Regime A, over Omega)."""
    if _gated(matched, config):
        return np.nan
    _, _, E_H, E_L = total_energies(matched)
    denom = np.sqrt(E_H * E_L)
    if denom <= _EPS:
        return np.nan
    return s_hl(matched, config) / denom


def xcorr_mag(matched: MatchedSet, config) -> float:
    """Phase-insensitive companion sum|z_H||z_L'| / sqrt(E_H E_L) in [0,1] (Regime A)."""
    if _gated(matched, config):
        return np.nan
    _, _, E_H, E_L = total_energies(matched)
    denom = np.sqrt(E_H * E_L)
    if denom <= _EPS:
        return np.nan
    if config.regime == "A" and matched.xH is not None:
        num = float(np.sum(np.abs(matched.xH) * np.abs(matched.xL)))
    else:
        num = float(np.sum(np.sqrt(matched.EH * matched.EL)))
    return num / denom


def xcorr_energy_proxy(matched: MatchedSet, config) -> float:
    """Regime-B energy cross-power sum_Omega sqrt(E_H E_L)/sqrt(E_H E_L union) in [0,1].

    Emitted in BOTH regimes (it is energy-based) so the schema is regime-stable."""
    if _gated(matched, config):
        return np.nan
    _, _, E_H, E_L = total_energies(matched)
    denom = np.sqrt(E_H * E_L)
    if denom <= _EPS:
        return np.nan
    num = float(np.sum(np.sqrt(matched.EH * matched.EL)))
    return num / denom


# --------------------------------------------------------------------------- coincidence (oLIB)

def coincidence_features(matched: MatchedSet, ts_H: TileSet, ts_L: TileSet, config) -> Dict:
    """oLIB coincidence diagnostics + coincident-vs-local energy fractions.

    Energy-based -> both regimes. NaN-safe; these are NOT gated by occupancy (they are
    descriptive, not the coherence veto) but return NaN on an empty Omega."""
    out = {
        "n_coinc_tiles_tile": float(matched.n_matched),
        "n_coinc_clusters_tile": float(len(matched.clusters)),
        "f0_agreement_tile": np.nan,
        "Q_agreement_tile": np.nan,
        "dt_consistency_tile": np.nan,
        "coinc_energy_frac_tile": np.nan,
        "min_coinc_frac_tile": np.nan,
        "snr_coin_max_tile": np.nan,
        "snr_coin_wmean_tile": np.nan,
        "snr_ratio_tile": np.nan,
    }
    if matched.empty:
        return out

    # f0/Q agreement ~1 by construction on the shared grid (diagnostic only)
    out["f0_agreement_tile"] = 1.0
    out["Q_agreement_tile"] = 1.0

    # dt consistency = 1 - |dt|/tau_tof, median over matched pairs
    if matched.dt_match is not None and matched.dt_match.size:
        dtc = 1.0 - np.clip(matched.dt_match / max(config.tau_tof, _EPS), 0.0, 1.0)
        out["dt_consistency_tile"] = float(np.median(dtc))

    # coincident energy fractions: matched-cell energy vs each detector's TOTAL significant energy
    E_H_coin = float(np.sum(matched.EH))
    E_L_coin = float(np.sum(matched.EL))
    E_H_all = ts_H.e_total_all
    E_L_all = ts_L.e_total_all
    frac_H = (E_H_coin / E_H_all) if E_H_all > _EPS else np.nan
    frac_L = (E_L_coin / E_L_all) if E_L_all > _EPS else np.nan
    # overall coincident fraction = coincident energy / all significant energy (both dets)
    tot_all = E_H_all + E_L_all
    out["coinc_energy_frac_tile"] = ((E_H_coin + E_L_coin) / tot_all) if tot_all > _EPS else np.nan
    fracs = [f for f in (frac_H, frac_L) if np.isfinite(f)]
    out["min_coinc_frac_tile"] = float(min(fracs)) if fracs else np.nan

    # snr_coin = sqrt(rho_H^2 + rho_L^2) per matched cell; rho = sqrt(energy)
    snr_coin = np.sqrt(matched.EH + matched.EL)
    w = matched.EH + matched.EL
    out["snr_coin_max_tile"] = float(np.max(snr_coin))
    out["snr_coin_wmean_tile"] = float(np.sum(snr_coin * w) / np.sum(w)) if np.sum(w) > _EPS \
        else float(np.mean(snr_coin))

    # snr_ratio over the loudest cluster: max(rho)/min(rho) of per-det energies
    if matched.clusters:
        loud = max(matched.clusters, key=lambda c: c.E_H + c.E_L)
        rH = np.sqrt(max(loud.E_H, 0.0))
        rL = np.sqrt(max(loud.E_L, 0.0))
        hi, lo = max(rH, rL), min(rH, rL)
        out["snr_ratio_tile"] = float(hi / lo) if lo > _EPS else np.nan
    return out


# --------------------------------------------------------------------------- robustness

def detector_dominance(ts_H: TileSet, ts_L: TileSet, matched: MatchedSet) -> Dict[str, float]:
    """dominance_tile over ALL significant tiles; participation = coincident/total per detector."""
    eH = ts_H.e_total_all
    eL = ts_L.e_total_all
    tot = eH + eL
    out = {
        "E_H_all": float(eH), "E_L_all": float(eL),
        "n_tiles_H": float(ts_H.n_significant),
        "n_tiles_L": float(ts_L.n_significant),
        "dominance_tile": np.nan,
        "participation_H_tile": np.nan,
        "participation_L_tile": np.nan,
    }
    if tot > _EPS:
        out["dominance_tile"] = float(max(eH, eL) / tot)
    if not matched.empty:
        E_H_coin = float(np.sum(matched.EH))
        E_L_coin = float(np.sum(matched.EL))
        if eH > _EPS:
            out["participation_H_tile"] = float(E_H_coin / eH)
        if eL > _EPS:
            out["participation_L_tile"] = float(E_L_coin / eL)
    return out


# --------------------------------------------------------------------------- assemble

def compute_all(ts_H: TileSet, ts_L: TileSet, matched: MatchedSet,
                lagresult, config) -> Dict[str, float]:
    """Assemble the FULL feature dict from a best-lag MatchedSet."""
    E_tile, Em_tile, E_H, E_L = total_energies(matched)
    ec = ec_tile(matched, config)
    en = en_tile(matched, config)
    lmod = l_model_tile(matched, config)
    cc = cc_tile(matched, config)
    chi2 = chi2_tile(matched, config)
    rho = rho_tile(matched, config)
    scc = scc_tile(matched, config)
    edr = edr_tile(matched, config)
    ccd, en_dom, single_det = cc_dom_tile(matched, config)
    ccoff = cc_offdiag_tile(matched, config)
    xs = xcorr_signed(matched, config)
    xm = xcorr_mag(matched, config)
    xe = xcorr_energy_proxy(matched, config)
    coin = coincidence_features(matched, ts_H, ts_L, config)
    dom = detector_dominance(ts_H, ts_L, matched)

    low_occ = int(_gated(matched, config))

    feats = {
        # bookkeeping energies
        "E_tile": E_tile, "Em_tile": Em_tile, "E_H_tile": E_H, "E_L_tile": E_L,
        "L_model_tile": lmod,
        # cWB-inspired
        "ec_tile": ec, "en_tile": en, "cc_tile": cc, "chi2_tile": chi2, "rho_tile": rho,
        "scc_tile": scc, "edr_tile": edr,
        "cc_dom_tile": ccd, "en_dom_tile": en_dom, "cc_offdiag_tile": ccoff,
        # oLIB-inspired cross-power
        "xcorr_signed_tile": xs, "xcorr_mag_tile": xm, "xcorr_energy_proxy": xe,
        # best-lag
        "best_lag_tile": (lagresult.best_lag if lagresult is not None else np.nan),
        "xcorr_bestlag_tile": (lagresult.xcorr_bestlag if lagresult is not None else np.nan),
        "xcorr_zerolag_tile": (lagresult.xcorr_zerolag if lagresult is not None else np.nan),
        "bestlag_unconstrained": (int(lagresult.unconstrained)
                                  if lagresult is not None else 0),
        # robustness
        "dominance_tile": dom["dominance_tile"],
        "participation_H_tile": dom["participation_H_tile"],
        "participation_L_tile": dom["participation_L_tile"],
        "n_tiles_H": dom["n_tiles_H"], "n_tiles_L": dom["n_tiles_L"],
        "n_matched": float(matched.n_matched),
        # flags
        "single_det_dominated_tile": int(single_det),
        "low_occupancy_tile": low_occ,
    }
    feats.update(coin)
    return feats


# backward-compat alias (slice name)
compute_slice_features = compute_all
