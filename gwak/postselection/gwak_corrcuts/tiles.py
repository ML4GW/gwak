"""tiles.py — TF decomposition on a shared gwpy QTiling grid.

THE CRUX. Regime A (default) replicates gwpy QTile.transform's per-tile windowed-FFT -> IFFT
(qtransform.py:427-435) and KEEPS the complex coefficient `cenergy` (gwpy squares-and-discards
it). Per row we normalize the complex coefficient by sqrt(median(|cenergy|^2)) so that the
normalized energy |x|^2 matches gwpy's norm='median' energy and E[|x|^2] ~= 1 on whitened noise.

Regime B is the stock energy-only path (plane.transform energy), kept for cross-check.

We do NOT re-whiten: the data is already whitened + std-normalized.

NOTE (verified against gwpy 2.1.3 source, deviation logged in the build report):
  * QTiling auto-truncates frange[1] to a Q-dependent max (~1291 Hz for qrange low=4); the grid
    we build reflects the truncated range. This is gwpy enforcing Nyquist/Q validity, not a bug.
  * QTile API used: .frequency, .q, .bandwidth, .ntiles, .duration, .get_data_indices(),
    .get_window(), .padding. All confirmed present in 2.1.3.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy import fft as npfft


# ----------------------------------------------------------------------------- grid

@dataclass
class GridRow:
    """One (plane, q, f0) row of the shared tiling. Time bins span [0, duration)."""
    plane_idx: int
    row_idx: int
    q: float
    f0: float
    duration: float          # tile time extent sigma_t ~ Q/(2 pi f0) (s)
    bandwidth: float         # tile bandwidth (Hz)
    ntiles: int              # number of time samples in this row's IFFT
    data_indices: np.ndarray  # int indices into the (one-sided) FFT
    window: np.ndarray        # bi-square window
    padding: Tuple[int, int]


@dataclass
class QGrid:
    qrange: Tuple[float, float]
    frange_effective: Tuple[float, float]   # AFTER gwpy truncation
    mismatch: float
    duration: float
    sampling: float
    rows: List[GridRow]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def signature(self) -> tuple:
        """Hashable identity for the shared-grid byte-identity test."""
        return tuple(
            (r.plane_idx, r.row_idx, round(r.q, 9), round(r.f0, 9), r.ntiles,
             int(r.data_indices[0]), int(r.data_indices[-1]), r.padding)
            for r in self.rows
        )


def build_grid(config) -> QGrid:
    """Construct ONE gwpy QTiling and enumerate it into a deterministic QGrid.

    Built once, shared by both detectors -> identical (plane, q, f0_row) lattice.
    """
    from gwpy.signal.qtransform import QTiling

    qt = QTiling(config.duration, config.sampling,
                 qrange=config.qrange, frange=config.frange, mismatch=config.mismatch)
    rows: List[GridRow] = []
    for pi, plane in enumerate(qt):
        for ri, tile in enumerate(plane):
            rows.append(GridRow(
                plane_idx=pi,
                row_idx=ri,
                q=float(tile.q),
                f0=float(tile.frequency),
                duration=float(tile.q / (2 * np.pi * tile.frequency)),
                bandwidth=float(tile.bandwidth),
                ntiles=int(tile.ntiles),
                data_indices=np.asarray(tile.get_data_indices(), dtype=np.int64),
                window=np.asarray(tile.get_window(), dtype=np.float64),
                padding=(int(tile.padding[0]), int(tile.padding[1])),
            ))
    return QGrid(
        qrange=(float(qt.qrange[0]), float(qt.qrange[1])),
        frange_effective=(float(qt.frange[0]), float(qt.frange[1])),
        mismatch=float(qt.mismatch),
        duration=float(config.duration),
        sampling=float(config.sampling),
        rows=rows,
    )


# ----------------------------------------------------------------------------- transform

@dataclass
class DetectorTransform:
    """Per-row aligned complex coefficients + energies for one detector."""
    detector: str
    status: str                      # ok | short | nan | missing
    # ragged per-row arrays (len = n_rows); each entry has length row.ntiles
    complex_rows: List[Optional[np.ndarray]]   # c16 (Regime A) else None
    energy_rows: List[np.ndarray]              # f8 normalized |x|^2 (E[.]~=1 noise)
    median_rows: np.ndarray                    # f8 per-row raw median(|cenergy|^2)
    grid: QGrid


def _rfft(x: np.ndarray, fs: float) -> np.ndarray:
    """One-sided FFT matching gwpy TimeSeries.fft() length/normalization.

    gwpy ts.fft() returns an rfft scaled by 1/n and with the non-DC, non-Nyquist
    bins doubled (one-sided). We reproduce that exactly so the data_indices line up.
    """
    n = x.size
    dft = npfft.rfft(x) / n
    # double the non-DC, non-Nyquist components (gwpy one-sided convention)
    if dft.size > 2:
        dft[1:-1] *= 2.0
    return dft


def transform_detector(x: np.ndarray, grid: QGrid, config, detector: str) -> DetectorTransform:
    """Decompose one whitened 1 s series onto the shared grid.

    Regime A: keep complex cenergy per tile, normalized by sqrt(median(|cenergy|^2)).
    Regime B: energy only (|cenergy|^2 / median), complex_rows = None.
    """
    n_rows = grid.n_rows

    # --- input validation / failure handling (NaN-safe) ---
    if x is None or x.size == 0:
        return DetectorTransform(detector, "missing", [None] * n_rows,
                                 [np.zeros(0)] * n_rows, np.full(n_rows, np.nan), grid)
    if x.size != config.n_samples:
        return DetectorTransform(detector, "short", [None] * n_rows,
                                 [np.zeros(0)] * n_rows, np.full(n_rows, np.nan), grid)

    status = "ok"
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        status = "nan"
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    fval = _rfft(x, config.sampling)

    regime_A = (config.regime == "A")
    complex_rows: List[Optional[np.ndarray]] = []
    energy_rows: List[np.ndarray] = []
    median_rows = np.empty(n_rows, dtype=np.float64)

    for i, row in enumerate(grid.rows):
        windowed = fval[row.data_indices] * row.window
        padded = np.pad(windowed, row.padding, mode="constant")
        wenergy = npfft.ifftshift(padded)
        cenergy = npfft.ifft(wenergy)                       # COMPLEX, kept
        raw_energy = cenergy.real ** 2 + cenergy.imag ** 2
        m = np.median(raw_energy)
        median_rows[i] = m
        if m <= 0 or not np.isfinite(m):
            # degenerate row (all-zero window region); avoid div-by-zero
            energy_rows.append(np.zeros_like(raw_energy))
            complex_rows.append(np.zeros_like(cenergy) if regime_A else None)
            continue
        norm_energy = raw_energy / m
        energy_rows.append(norm_energy)
        if regime_A:
            # amplitude normalization so |x|^2 == norm_energy
            complex_rows.append(cenergy / np.sqrt(m))
        else:
            complex_rows.append(None)

    return DetectorTransform(detector, status, complex_rows, energy_rows, median_rows, grid)


# ----------------------------------------------------------------------------- tiles / TileSet

@dataclass
class Tile:
    row_idx: int          # index into grid.rows -> identifies (plane, q, f0)
    time_bin: int         # index within the row's ntiles time samples
    t: float              # center time (s)
    f0: float
    q: float
    duration: float
    bandwidth: float
    energy: float         # normalized |x|^2
    complex_amp: Optional[complex]
    detector: str


@dataclass
class TileSet:
    detector: str
    status: str
    tiles: List[Tile]
    dt: DetectorTransform           # full transform (for matching by row/time)
    grid: QGrid
    n_significant: int
    e_total_all: float              # total normalized energy over ALL significant tiles

    @property
    def regime_label(self) -> str:
        return "A_complex_faithful" if self.dt.complex_rows and \
            any(c is not None for c in self.dt.complex_rows) else "B_unsigned_degraded"


def extract_tiles(x: np.ndarray, grid: QGrid, config, detector: str) -> TileSet:
    """Transform then apply the significance threshold; build Tile records. NaN-safe."""
    dt = transform_detector(x, grid, config, detector)
    if dt.status in ("missing", "short", "nan") and dt.status != "ok":
        # still build (possibly empty) tiles for nan (it computed); missing/short -> empty
        if dt.status in ("missing", "short"):
            return TileSet(detector, dt.status, [], dt, grid, 0, 0.0)

    thr = config.sig_energy_thr if config.e_thr_calibrated is None else config.e_thr_calibrated
    tiles: List[Tile] = []
    e_total_all = 0.0
    for ri, row in enumerate(grid.rows):
        e_row = dt.energy_rows[ri]
        if e_row.size == 0:
            continue
        nt = row.ntiles
        # time of bin j: center within [0, duration)
        dt_bin = grid.duration / nt
        sig = np.where(e_row >= thr)[0]
        for j in sig:
            energy = float(e_row[j])
            e_total_all += energy
            camp = None
            if dt.complex_rows[ri] is not None:
                camp = complex(dt.complex_rows[ri][j])
            tiles.append(Tile(
                row_idx=ri, time_bin=int(j),
                t=float((j + 0.5) * dt_bin),
                f0=row.f0, q=row.q, duration=row.duration, bandwidth=row.bandwidth,
                energy=energy, complex_amp=camp, detector=detector,
            ))
    return TileSet(detector, dt.status, tiles, dt, grid, len(tiles), e_total_all)
