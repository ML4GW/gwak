"""calibrate.py — data-driven calibration of E_thr and kappa_noise on the Background class.

The data is per-detector unit-variance normalized, so the absolute energy scale is gone. Both
the significance threshold E_thr (DESIGN §3 / features_reconciled eqn 14) and the chi2 noise
constant kappa_noise (DESIGN §5 eqn 12) are therefore CALIBRATED on the Background class, not
imported from oLIB/Omicron.

E_thr: a fixed background tile-energy quantile. With median-normalized energy ~ Exp(median=1)
on noise, a high quantile (default 0.999 over all tiles) sets a per-tile occupancy target.

kappa_noise: with E_thr fixed, run the matcher on Background, collect en_tile/|Omega| over events
with a populated occupancy gate, and set kappa_noise = mean(en_tile/|Omega|) so that
E[chi2_tile] = E[en/(kappa*|Omega|)] ~= 1 on Background.
"""
import numpy as np

from . import io as _io
from .matching import best_lag_scan, match_tiles
from .tiles import build_grid, extract_tiles
from .features import en_tile, occupancy_ok


def _collect_background_energies(path, config, grid, n_events, rng):
    """Return a flat array of per-row normalized tile energies over a Background subsample."""
    energies = []
    seen = 0
    for batch in _io.iter_event_batches(path, "Background", config.batch_size,
                                        subsample=n_events, rng=rng):
        for i in range(batch.data.shape[0]):
            for det, ci in (("H1", 0), ("L1", 1)):
                from .tiles import transform_detector
                dt = transform_detector(batch.data[i, ci, :], grid, config, det)
                if dt.status != "ok":
                    continue
                for er in dt.energy_rows:
                    if er.size:
                        energies.append(er)
            seen += 1
            if seen >= n_events:
                break
        if seen >= n_events:
            break
    if not energies:
        return np.zeros(0)
    return np.concatenate(energies)


def calibrate(path, config, n_events_ethr=200, ethr_quantile=0.999,
              n_events_kappa=400, seed=None):
    """Calibrate E_thr and kappa_noise on Background. Mutates a COPY of config is NOT done;
    instead returns (e_thr, kappa_noise) and sets them on the passed config in place.

    Steps:
      1. E_thr = background tile-energy `ethr_quantile` quantile (over many noise tiles).
      2. With E_thr set, run the full matcher on a fresh Background subsample; collect
         en_tile/|Omega| over occupancy-gated events; kappa_noise = mean of those.
    Returns dict with e_thr, kappa_noise, e_chi2_check (held-out E[chi2_tile]), and counts.
    """
    if seed is None:
        seed = config.seed
    grid = build_grid(config)

    # --- step 1: E_thr from a background tile-energy quantile ---
    rng1 = np.random.RandomState(seed)
    bg_energies = _collect_background_energies(path, config, grid, n_events_ethr, rng1)
    if bg_energies.size == 0:
        raise RuntimeError("no Background tile energies collected for E_thr calibration")
    e_thr = float(np.quantile(bg_energies, ethr_quantile))
    config.e_thr_calibrated = e_thr

    # --- step 2: kappa_noise so E[chi2]~=1 on Background ---
    rng2 = np.random.RandomState(seed + 1)
    per_event = []   # en/|Omega| per occupancy-gated event
    n_proc = 0
    n_gated_ok = 0
    for batch in _io.iter_event_batches(path, "Background", config.batch_size,
                                        subsample=n_events_kappa, rng=rng2):
        for i in range(batch.data.shape[0]):
            xH = batch.data[i, 0, :]
            xL = batch.data[i, 1, :]
            ts_H = extract_tiles(xH, grid, config, "H1")
            ts_L = extract_tiles(xL, grid, config, "L1")
            n_proc += 1
            if ts_H.status != "ok" or ts_L.status != "ok":
                continue
            lag = best_lag_scan(ts_H, ts_L, config)
            bl = lag.best_lag if np.isfinite(lag.best_lag) else 0.0
            m = match_tiles(ts_H, ts_L, config, lag=bl)
            if not occupancy_ok(m, config):
                continue
            # en_tile computed with kappa=None internally is just the null-stream energy
            en = en_tile(m, config)
            if np.isfinite(en) and m.n_matched > 0:
                per_event.append(en / m.n_matched)
                n_gated_ok += 1

    if not per_event:
        # No occupancy-gated background events at this E_thr: kappa cannot be fit.
        config.kappa_noise = None
        return {
            "e_thr": e_thr, "ethr_quantile": ethr_quantile,
            "kappa_noise": None, "e_chi2_check": None,
            "n_bg_tiles_ethr": int(bg_energies.size),
            "n_events_processed_kappa": n_proc, "n_gated_ok": 0,
            "note": "no occupancy-gated Background events at this E_thr; kappa_noise unset",
        }

    per_event = np.asarray(per_event, dtype=np.float64)
    kappa_noise = float(np.mean(per_event))
    config.kappa_noise = kappa_noise

    # --- held-out E[chi2] check on a fresh subsample ---
    rng3 = np.random.RandomState(seed + 2)
    chi2_vals = []
    from .features import chi2_tile
    for batch in _io.iter_event_batches(path, "Background", config.batch_size,
                                        subsample=n_events_kappa, rng=rng3):
        for i in range(batch.data.shape[0]):
            ts_H = extract_tiles(batch.data[i, 0, :], grid, config, "H1")
            ts_L = extract_tiles(batch.data[i, 1, :], grid, config, "L1")
            if ts_H.status != "ok" or ts_L.status != "ok":
                continue
            lag = best_lag_scan(ts_H, ts_L, config)
            bl = lag.best_lag if np.isfinite(lag.best_lag) else 0.0
            m = match_tiles(ts_H, ts_L, config, lag=bl)
            c2 = chi2_tile(m, config)
            if np.isfinite(c2):
                chi2_vals.append(c2)
    e_chi2 = float(np.mean(chi2_vals)) if chi2_vals else None

    return {
        "e_thr": e_thr, "ethr_quantile": ethr_quantile,
        "kappa_noise": kappa_noise, "e_chi2_check": e_chi2,
        "n_bg_tiles_ethr": int(bg_energies.size),
        "n_events_processed_kappa": n_proc, "n_gated_ok": n_gated_ok,
        "n_chi2_heldout": len(chi2_vals),
    }
