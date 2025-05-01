#!/usr/bin/env python3
"""
Save FAR metrics from streamed-inference HDF5 files **and** draw the cumulative
background-events vs iFAR plot **with Poisson 1 / 2 / 3 σ bands**.

Key fixes
~~~~~~~~~
* **iFAR axis now ascends** (small on the left, large on the right).
* Predicted curve therefore slopes **downward** left→right.
* Confidence bands follow the same orientation.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

SECONDS_PER_YEAR: float = 365.0 * 24 * 3_600  # 31 536 000 s

# -----------------------------------------------------------------------------
# I/O helpers -----------------------------------------------------------------

def merge_h5_files(folder_path: str, dataset_key: str = "data") -> np.ndarray:
    """Concatenate every ``*.h5`` in *folder_path* along axis-0."""
    files = sorted(glob.glob(os.path.join(folder_path, "*.h5")))
    if not files:
        raise ValueError(f"No .h5 files found in {folder_path!r}.")

    parts: list[np.ndarray] = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            data = f[dataset_key][:]
            parts.append(data)
            print(f"  loaded {data.shape} from {os.path.basename(fn)}")

    merged = np.concatenate(parts, axis=0)
    print(f"Merged array shape: {merged.shape}")
    return merged

# -----------------------------------------------------------------------------
# FAR helpers -----------------------------------------------------------------

def score_to_far(threshold: float, scores: np.ndarray, total_time: float, *,
                 direction: str = "negative") -> float:
    """Return FAR (events / second) for *threshold* against *scores*."""
    if direction == "negative":
        n = np.count_nonzero(scores <= threshold)
    else:
        n = np.count_nonzero(scores >= threshold)
    return n / total_time

def calc_far_table(merged: np.ndarray, duration: float, *, num: int = 100,
                   direction: str = "negative") -> Tuple[np.ndarray, float, np.ndarray]:
    """Uniform *num*-point score→FAR table."""
    n, m, _, _ = merged.shape
    total_time = n * m * duration  # seconds

    scores = merged.reshape(-1)
    thresholds = np.linspace(scores.min(), scores.max(), num)
    fars = np.array([score_to_far(t, scores, total_time, direction=direction)
                     for t in thresholds])
    return np.column_stack([thresholds, fars]), total_time, scores

# -----------------------------------------------------------------------------
# Plot helper -----------------------------------------------------------------

def plot_ifar_cumulative(scores: np.ndarray,
                         total_time: float,
                         *,
                         direction: str = "negative",
                         outfile: str = "ifar_vs_cumulative_background.png",
                         sigma_alphas: tuple[float, float, float] = (0.4, 0.25, 0.1),
                         max_points: int | None = 5000) -> None:
    """Cumulative #events vs iFAR **with** Poisson 68 / 95 / 99.7 % bands.

    *Axis conventions*
    ------------------
    • x-axis (iFAR) **ascends** left→right (0.01 yr … 100 yr).
    • Predicted curve ``N = T_live / iFAR`` therefore slopes **down**.
    """
    scores = np.asarray(scores)

    # 1. Rank statistics --------------------------------------------------
    if direction == "negative":
        order = np.argsort(scores)          # ascending (most-negative last)
    else:
        order = np.argsort(scores)[::-1]

    # 1.  Order scores once (ascending if "negative")
    scores_sorted = np.sort(scores) if direction == "negative" else np.sort(scores)[::-1]

    # 2.  For every original score find how many sorted scores are <= that score
    #     (or >= if direction is "positive")
    trigger_counts = np.searchsorted(scores_sorted, scores, side="right")  # vectorised

    far_sec = trigger_counts / total_time        # events / second
    ifar_years = 1.0 / (far_sec * SECONDS_PER_YEAR)

    idx         = np.argsort(ifar_years)               # iFAR small→large
    ifar_sorted = ifar_years[idx]
    cum_sorted  = trigger_counts[idx]                  # <-- use the real counts


    # Optional thinning ---------------------------------------------------
    # if max_points and len(ifar_sorted) > max_points:
    #    step = len(ifar_sorted) // max_points
    #    ifar_sorted = ifar_sorted[::step]
    #    cum_sorted = cum_sorted[::step]
    # --- optional density reduction for plotting --------------------------
    if max_points and len(ifar_sorted) > max_points:
        import math
        # cum_sorted is DESCENDING (N … 1).  Build the target levels.
        c_max = cum_sorted[0]
        exp_max = int(math.log10(c_max))           # e.g. 7 → for 10^7
        levels = []
        for e in range(exp_max, -1, -1):           # 10^e … 10^0
            base = 10 ** e
            for k in range(10, 0, -1):             # 10·base .. 1·base
                lev = k * base
                if lev <= c_max and lev >= 1:
                    levels.append(lev)
        levels = np.array(levels)

        # first index where cum_sorted drops to ≤ level (because cum_sorted desc)
        keep_idx = np.searchsorted(cum_sorted[::-1], levels, side='left')
        keep_idx = len(cum_sorted) - 1 - keep_idx      # map back to forward indices

        # add the two endpoints and make indices unique / sorted
        keep_idx = np.unique(np.r_[0, keep_idx, len(cum_sorted) - 1])

        ifar_sorted = ifar_sorted[keep_idx]
        cum_sorted  = cum_sorted[keep_idx]
            
    # 2. Predicted curve & Poisson bands ----------------------------------
    livetime_yr = total_time / SECONDS_PER_YEAR

    y_max = max(cum_sorted) * 1.2
    y_min = 1e-3
    predicted_y = np.logspace(np.log10(y_max), np.log10(y_min), 1000)  # decreasing
    predicted_x = livetime_yr / predicted_y                             # increasing

    fap1, fap2, fap3 = 0.682689, 0.954499, 0.997300
    sig1 = np.array(poisson.interval(fap1, predicted_y))
    sig2 = np.array(poisson.interval(fap2, predicted_y))
    sig3 = np.array(poisson.interval(fap3, predicted_y))
    for arr in (sig1, sig2, sig3):
        arr[0] = np.maximum(arr[0] - 0.5, 0.0)
        arr[1] = arr[1] + 0.5

    # 3. Plot --------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))

    plt.fill_between(predicted_x, sig3[0], sig3[1], color="steelblue", alpha=sigma_alphas[2], label=r"1,2,3 $\sigma$")
    plt.fill_between(predicted_x, sig2[0], sig2[1], color="steelblue", alpha=sigma_alphas[1])
    plt.fill_between(predicted_x, sig1[0], sig1[1], color="steelblue", alpha=sigma_alphas[0])

    plt.plot(predicted_x, predicted_y, color="steelblue", lw=2, label="Predicted")
    plt.step(ifar_sorted, cum_sorted, where="post", lw=2, marker="o", ms=3,
             color="#f29e4c", label="Background")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("iFAR (years)")
    plt.ylabel("Cumulative number of events")
    plt.title("Background cumulative distribution vs iFAR")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlim(1e-4, ifar_sorted.max()*100)
    plt.ylim(0.9,1e4)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved plot → {outfile}")

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge HDF5 scores, compute FAR table, and plot iFAR distribution with Poisson bands.")
    ap.add_argument("folder", help="Folder containing *.h5 score files")
    ap.add_argument("--dataset", default="data")
    ap.add_argument("--duration", type=float, default=0.5, help="Duration per score [s]")
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--outfile", default="far_metrics.npy")
    ap.add_argument("--direction", choices=["negative", "positive"], default="")
    ap.add_argument("--plotfile", default="ifar_vs_cumulative_background.png")
    ap.add_argument("--max-points", type=int, default=50_000)
    args = ap.parse_args()

    merged = merge_h5_files(args.folder, args.dataset)

    far_table, total_time_sec, scores = calc_far_table(
        merged, args.duration, num=args.num_thresholds, direction=args.direction)
    np.save(args.outfile, far_table)
    print(f"Saved FAR table → {args.outfile}")

    print(scores)
    print(len(scores))

    # create random scores of the same shape as scores to test

    scores = np.random.normal(size=scores.shape)
    #scores = np.ones_like(scores)
    #print(scores)
    #print(len(scores))

    plot_ifar_cumulative(scores, total_time_sec,
                         direction=args.direction,
                         outfile=args.plotfile,
                         max_points=args.max_points)

if __name__ == "__main__":
    main()
