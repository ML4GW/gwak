"""config.py — the Config dataclass (all tunables) + to_dict/from_dict.

No physics here, just parameters and validation. Defaults are the DESIGN.md values.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional


@dataclass
class Config:
    # --- Q-transform / tile grid (DESIGN §3) ---
    qrange: Tuple[float, float] = (4.0, 64.0)
    frange: Tuple[float, float] = (32.0, 2000.0)   # gwpy may auto-truncate the upper edge
    mismatch: float = 0.20
    sampling: float = 4096.0                       # fs; asserted against data last axis
    duration: float = 1.0                          # s; asserted against data last axis

    # --- regime (DESIGN §2) ---
    # "A" = A_complex_faithful (default, signed). "B" = B_unsigned_degraded (energy-only).
    regime: str = "A"
    allow_regime_fallback: bool = False

    # --- significant-tile selection (DESIGN §3 / features_reconciled §6 eqn 14) ---
    # E_thr is calibrated on Background in the full build. For the slice we use a fixed
    # default keep-fraction proxy: a tile is significant iff |x|^2 >= sig_energy_thr.
    # With median-normalization E[|x|^2] ~= 1 on noise, so sig_energy_thr > 1 removes the floor.
    # With median-normalized energy ~ Exp(median=1) on noise, P(E>=t) ~= 2^-t, so thr=12
    # leaves ~2e-4 noise occupancy per tile while keeping loud (SNR>>4) signal tiles. Calibrated
    # E_thr on Background replaces this in the full build (slice default is a fixed proxy).
    sig_energy_thr: float = 12.0    # keep tiles with normalized energy >= 12
    e_thr_calibrated: Optional[float] = None   # set when calibrated on Background
    # Minimum matched-cell occupancy for the coherence features to be defined. With < min_omega
    # cells the signed cross-term is trivially "coherent" (single-pair degeneracy), so cc_tile et
    # al. are NaN below this (an occupancy gate, NOT a cut on the feature value). Independent
    # noise produces ~0 coincident significant tiles, so it lands NaN here -> correct (no false
    # coherent accept), distinct from a populated-but-incoherent small cc.
    min_omega: int = 3
    # A "cluster" of one tile is not a cluster; the occupancy gate additionally requires >=1
    # matched coincident cluster with >= k_cluster tiles (cWB min-pixel / oLIB coincident-
    # cluster requirement; ruling Q2).
    k_cluster: int = 2

    # --- matching / time-of-flight / best-lag (DESIGN §4) ---
    shared_grid: bool = True
    tau_tof: float = 0.010          # H1-L1 light travel ~10 ms; coincidence band (one-sided)
    lag_window: float = 0.010       # best-lag scan range +/- 10 ms
    lag_step: float = 0.001         # 1 ms step
    dt_cluster: float = 0.100       # cluster matched cells within 100 ms
    # best-lag envelope-contrast threshold (ruling Q1): when the carrier-free envelope contrast
    # (max|S_HL| - median|S_HL|)/max|S_HL| < contrast_min the lag is physically unresolvable
    # (narrowband), so bestlag_unconstrained=1 and coherence is read at zero-lag.
    contrast_min: float = 0.4

    # --- feature regulators ---
    n_reg: float = 1.0              # scc_tile O(1) regulator (NOT the tile count)
    c_dof: int = 1                  # K-1 for K=2
    K: int = 2
    # chi2 noise normalization constant; calibrated on Background so E[chi2_tile] ~= 1.
    # None -> chi2_tile falls back to en/|Omega| (uncalibrated) and is flagged in metadata.
    kappa_noise: Optional[float] = None

    # --- plotting / validation ---
    snr_bin_edges: Tuple[float, ...] = (3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 25.0, 30.0)
    glitch_role: str = "background"

    # --- I/O / streaming ---
    batch_size: int = 256
    subsample: Optional[int] = None       # deterministic per-class subsample for dev
    seed: int = 1234
    output_format: str = "csv"            # slice: csv (parquet optional)
    also_csv: bool = True

    def __post_init__(self):
        if self.regime not in ("A", "B"):
            raise ValueError("regime must be 'A' or 'B', got %r" % (self.regime,))
        if self.qrange[0] <= 0 or self.qrange[1] <= self.qrange[0]:
            raise ValueError("invalid qrange %r" % (self.qrange,))
        if self.frange[0] <= 0 or self.frange[1] <= self.frange[0]:
            raise ValueError("invalid frange %r" % (self.frange,))
        if not (0 < self.mismatch < 1):
            raise ValueError("invalid mismatch %r" % (self.mismatch,))
        # normalize tuples (JSON round-trips lists)
        self.qrange = (float(self.qrange[0]), float(self.qrange[1]))
        self.frange = (float(self.frange[0]), float(self.frange[1]))

    @property
    def regime_label(self) -> str:
        return "A_complex_faithful" if self.regime == "A" else "B_unsigned_degraded"

    @property
    def n_samples(self) -> int:
        return int(round(self.sampling * self.duration))

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuples -> lists for JSON
        d["qrange"] = list(self.qrange)
        d["frange"] = list(self.frange)
        d["snr_bin_edges"] = list(self.snr_bin_edges)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        if "qrange" in d:
            d["qrange"] = tuple(d["qrange"])
        if "frange" in d:
            d["frange"] = tuple(d["frange"])
        if "snr_bin_edges" in d:
            d["snr_bin_edges"] = tuple(d["snr_bin_edges"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)
