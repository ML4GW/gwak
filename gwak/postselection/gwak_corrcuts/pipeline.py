"""pipeline.py — orchestration: process_event + run over a subsample.

process_event wraps each stage in try/except -> status/fail_reason instead of crashing.
"""
from typing import List, Optional

import numpy as np

from . import io as _io
from .features import compute_all
from .matching import best_lag_scan, match_tiles
from .tiles import build_grid, extract_tiles


# The full feature columns emitted (for a stable schema / NaN fill on failure).
_FEATURE_KEYS = [
    # bookkeeping energies
    "E_tile", "Em_tile", "E_H_tile", "E_L_tile", "L_model_tile",
    # cWB-inspired
    "ec_tile", "en_tile", "cc_tile", "chi2_tile", "rho_tile", "scc_tile", "edr_tile",
    "cc_dom_tile", "en_dom_tile", "cc_offdiag_tile",
    # oLIB cross-power
    "xcorr_signed_tile", "xcorr_mag_tile", "xcorr_energy_proxy",
    # best-lag
    "best_lag_tile", "xcorr_bestlag_tile", "xcorr_zerolag_tile", "bestlag_unconstrained",
    # robustness
    "dominance_tile", "participation_H_tile", "participation_L_tile",
    "n_tiles_H", "n_tiles_L", "n_matched",
    # flags
    "single_det_dominated_tile", "low_occupancy_tile",
    # coincidence (oLIB)
    "n_coinc_tiles_tile", "n_coinc_clusters_tile", "f0_agreement_tile", "Q_agreement_tile",
    "dt_consistency_tile", "coinc_energy_frac_tile", "min_coinc_frac_tile",
    "snr_coin_max_tile", "snr_coin_wmean_tile", "snr_ratio_tile",
]


def _nan_features():
    d = {k: np.nan for k in _FEATURE_KEYS}
    d["bestlag_unconstrained"] = 0
    d["single_det_dominated_tile"] = 1
    d["low_occupancy_tile"] = 1
    return d


def process_event(xH, xL, snr, class_name, row_index, grid, config) -> dict:
    """Single event: extract -> match -> best-lag -> features -> row dict (status-flagged)."""
    row = {
        "event_id": "%s:%d" % (class_name, int(row_index)),
        "class_name": class_name,
        "row_index": int(row_index),
        "snr": float(snr),
        "regime": config.regime_label,
        "status": "ok",
        "fail_reason": "",
        "match_mode": "shared_grid" if config.shared_grid else "fractional",
    }
    try:
        ts_H = extract_tiles(xH, grid, config, "H1")
        ts_L = extract_tiles(xL, grid, config, "L1")
        # determine event status from the worse of the two
        statuses = {ts_H.status, ts_L.status}
        if "missing" in statuses:
            row["status"] = "missing"
        elif "short" in statuses:
            row["status"] = "short"
        elif "nan" in statuses:
            row["status"] = "nan"

        if row["status"] in ("missing", "short"):
            row.update(_nan_features())
            row["fail_reason"] = "detector status=%s" % row["status"]
            return row

        lag = best_lag_scan(ts_H, ts_L, config)
        best = lag.best_lag if np.isfinite(lag.best_lag) else 0.0
        matched = match_tiles(ts_H, ts_L, config, lag=best)
        feats = compute_all(ts_H, ts_L, matched, lag, config)
        row.update(feats)
        return row
    except Exception as exc:  # noqa: BLE001 - never abort the run
        row["status"] = "transform_fail"
        row["fail_reason"] = "%s: %s" % (type(exc).__name__, exc)
        row.update(_nan_features())
        return row


def run(path: str, config, out_path: str,
        classes: Optional[List[str]] = None) -> List[dict]:
    """Stream classes/batches, process each event, write the table. Returns the rows."""
    grid = build_grid(config)
    if classes is None:
        classes = _io.list_classes(path)
    rng = np.random.RandomState(config.seed)
    rows: List[dict] = []
    for cls in classes:
        for batch in _io.iter_event_batches(path, cls, config.batch_size,
                                            subsample=config.subsample, rng=rng):
            b = batch.data.shape[0]
            for i in range(b):
                xH = batch.data[i, 0, :]
                xL = batch.data[i, 1, :]
                rows.append(process_event(xH, xL, batch.snr[i], cls,
                                          batch.row_indices[i], grid, config))
    _io.write_feature_table(rows, config, out_path)
    return rows
