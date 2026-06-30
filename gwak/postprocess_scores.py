#!/usr/bin/env python3
"""
Apply smoothing and/or veto to existing scores.npy and replot.

Usage:
    python postprocess_scores.py \
        --scores output/model_IF/evaluation/scores.npy \
        --output-dir output/model_IF/evaluation/ \
        --smooth-window 1 \
        --veto-duration 10
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--scores",        required=True, help="Path to existing scores.npy")
ap.add_argument("--output-dir",    required=True)
ap.add_argument("--smooth-window", type=int, default=1)
ap.add_argument("--veto-duration", type=int, default=0)
args = ap.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

scores = np.load(args.scores)
scores = scores[np.isfinite(scores)]
print(f"Loaded {len(scores):,} scores  [{scores.min():.4f}, {scores.max():.4f}]")

if args.smooth_window > 1:
    kernel = np.ones(args.smooth_window, dtype=np.float32) / args.smooth_window
    scores = np.convolve(scores, kernel, mode='same')
    print(f"Applied top-hat smoothing: window={args.smooth_window}")

if args.veto_duration > 0:
    flag_threshold = np.percentile(scores, 1)
    flagged = scores < flag_threshold
    keep = np.ones(len(scores), dtype=bool)
    i = 0
    while i < len(scores):
        if flagged[i]:
            j = i
            while j < len(scores) and flagged[j]:
                j += 1
            if (j - i) > args.veto_duration:
                keep[i:j] = False
            i = j
        else:
            i += 1
    print(f"10s veto: removed {(~keep).sum():,} chunks")
    scores = scores[keep]

scores_path = os.path.join(args.output_dir, "scores_postprocessed.npy")
np.save(scores_path, scores)
print(f"Saved postprocessed scores: {scores_path}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(scores, bins=300, color="darkorange", alpha=0.8)
ax.set_yscale("log")
ax.set_xlabel("IF Anomaly Score (lower = more anomalous)")
ax.set_ylabel("Count")
ax.set_title(f"Score Distribution (post-processed)\n"
             f"n={len(scores):,}  min={scores.min():.3f}  max={scores.max():.3f}")
ax.grid(True, which="both", ls="--", alpha=0.3)
fig.tight_layout()
out = os.path.join(args.output_dir, "score_histogram_postprocessed.png")
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"Histogram saved: {out}")


