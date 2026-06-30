#!/usr/bin/env python3
"""
Plot detection efficiency vs SNR and hrss for either:
  - Combined JIT model (NF mode), when only --model-path is given
  - Embedding model + Isolation Forest (IF mode), when --if-model is also given

Usage (NF):
    python efficiency_plots.py \
        --model-path output/model/combination/model_JIT.pt \
        --background-scores output/model/evaluation/scores.npy \
        --signal-dataset output/dataset_train_HL_SR4096_kernel1.0_hrss.h5 \
        --output-dir output/model/evaluation/

Usage (IF):
    python efficiency_plots.py \
        --model-path output/model/model_JIT.pt \
        --if-model output/model/isolation_forest.joblib \
        --means output/model/means.npy \
        --stds  output/model/stds.npy \
        --background-scores output/model_IF/evaluation/scores.npy \
        --signal-dataset output/dataset_train_HL_SR4096_kernel1.0_hrss.h5 \
        --output-dir output/model_IF/evaluation/
"""

import argparse
import os
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model-path",        required=True,
                    help="Combined JIT model (NF mode) or embedding JIT model (IF mode)")
parser.add_argument("--if-model",          default=None,
                    help="Path to isolation_forest.joblib — triggers IF mode")
parser.add_argument("--means",             default=None)
parser.add_argument("--stds",              default=None)
parser.add_argument("--background-scores", required=True)
parser.add_argument("--signal-dataset",    required=True)
parser.add_argument("--output-dir",        required=True)
parser.add_argument("--threshold",         type=float, default=None)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

SIGNAL_CLASSES     = ["BBH", "CCSN", "Cusp", "KinkKink", "Kink", "SineGaussian", "WhiteNoiseBurst"]
BACKGROUND_CLASSES = ["Background", "Glitch"]
ALL_CLASSES        = SIGNAL_CLASSES + BACKGROUND_CLASSES
BATCH_SIZE = 256
NUM_BINS   = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
mode = "if" if args.if_model else "nf"
print(f"Mode: {'Isolation Forest' if mode == 'if' else 'Normalizing Flow'}")

# ── 1. Threshold from background scores ────────────────────────────────────────
bg_scores = np.load(args.background_scores)
bg_scores = bg_scores[np.isfinite(bg_scores)]
threshold = float(args.threshold) if args.threshold is not None else float(bg_scores.min())
print(f"Background samples : {len(bg_scores):,}")
print(f"Score range        : [{bg_scores.min():.4f}, {bg_scores.max():.4f}]")
print(f"Threshold          : {threshold:.4f}")

score_label = "IF Anomaly Score (lower = more anomalous)" if mode == "if" else "GWAK Score"
bg_color    = "darkorange" if mode == "if" else "steelblue"

plt.figure(figsize=(8, 4))
plt.hist(bg_scores, bins=300, color=bg_color, alpha=0.8)
plt.axvline(threshold, color="red", lw=1.5, label=f"threshold = {threshold:.3f}")
plt.yscale("log")
plt.xlabel(score_label)
plt.ylabel("Count")
plt.title("Background Score Distribution — one month")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "background_scores.png"), dpi=200)
plt.close()
print("Saved background_scores.png")

# ── 2. Load models ──────────────────────────────────────────────────────────────
if mode == "if":
    import joblib
    print(f"\nLoading embedding model: {args.model_path}")
    embed_model = torch.jit.load(args.model_path, map_location=device)
    embed_model.eval()
    print(f"Loading Isolation Forest: {args.if_model}")
    clf   = joblib.load(args.if_model)
    means = np.load(args.means).astype(np.float32)
    stds  = np.load(args.stds).astype(np.float32)
else:
    print(f"\nLoading model: {args.model_path}")
    model = torch.jit.load(args.model_path, map_location=device)
    model.eval()
print("Model(s) loaded.\n")

# ── 3. Load signal dataset ──────────────────────────────────────────────────────
print("Loading signal dataset...")
data_by_class = {}
with h5py.File(args.signal_dataset, "r") as f:
    for cls in ALL_CLASSES:
        if f"{cls}_data" not in f:
            print(f"  [skip] {cls}: not found")
            continue
        data_by_class[cls] = {
            "data": f[f"{cls}_data"][:],
            "snrs": f[f"{cls}_snrs"][:],
            "hrss": f[f"{cls}_hrss"][:],
        }
        n, snr, h = (data_by_class[cls]["data"].shape[0],
                     data_by_class[cls]["snrs"], data_by_class[cls]["hrss"])
        print(f"  {cls:20s}: {n:>6,} samples  snr=[{snr.min():.1f},{snr.max():.1f}]  hrss=[{h.min():.2e},{h.max():.2e}]")

SIGNAL_CLASSES     = [c for c in SIGNAL_CLASSES     if c in data_by_class]
BACKGROUND_CLASSES = [c for c in BACKGROUND_CLASSES if c in data_by_class]

# ── 4. Score signals ────────────────────────────────────────────────────────────
print("\nScoring signals...")
scores_by_class = {}
for cls in data_by_class:
    chunks = data_by_class[cls]["data"].astype(np.float32)
    if mode == "if":
        cls_scores = []
        for start in range(0, len(chunks), BATCH_SIZE):
            x = torch.tensor(chunks[start:start + BATCH_SIZE]).to(device)
            with torch.no_grad():
                embeddings = embed_model(x).cpu().numpy()
            H = torch.fft.rfft(x[:, 0, :], dim=-1)
            L = torch.fft.rfft(x[:, 1, :], dim=-1)
            xcorr = torch.real(
                torch.sum(H * torch.conj(L), dim=-1) /
                (torch.linalg.norm(H, dim=-1) * torch.linalg.norm(L, dim=-1) + 1e-8)
            ).cpu().numpy().reshape(-1, 1)
            feats = np.concatenate([(embeddings - means) / (stds + 1e-8), xcorr], axis=1)
            cls_scores.append(clf.score_samples(feats).astype(np.float32))
        scores_by_class[cls] = np.concatenate(cls_scores)
    else:
        cls_scores = []
        for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc=cls, leave=False):
            x = torch.tensor(chunks[start:start + BATCH_SIZE]).to(device)
            with torch.no_grad():
                out = model(x)
            cls_scores.append(out.cpu().numpy().reshape(-1))
        scores_by_class[cls] = np.concatenate(cls_scores)
    s = scores_by_class[cls]
    print(f"  {cls:20s}: score=[{s.min():.3f}, {s.max():.3f}]  detected={np.mean(s < threshold)*100:.1f}%")

# ── 5. Score distributions (IF only) ───────────────────────────────────────────
if mode == "if":
    fig, ax = plt.subplots(figsize=(10, 6))
    score_min = min(bg_scores.min(), *(scores_by_class[c].min() for c in SIGNAL_CLASSES))
    score_max = max(bg_scores.max(), *(scores_by_class[c].max() for c in SIGNAL_CLASSES))
    bins = np.linspace(score_min, score_max, 200)
    ax.hist(bg_scores, bins=bins, color="gray", alpha=0.5, label="Background (1 month)", density=True)
    for cls in SIGNAL_CLASSES:
        ax.hist(scores_by_class[cls], bins=bins, alpha=0.6, label=cls, density=True, histtype="step", lw=1.5)
    ax.axvline(threshold, color="red", lw=1.5, ls="--", label=f"threshold = {threshold:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel(score_label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distributions — Background vs Signals", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "score_distributions.png"), dpi=200)
    plt.close()
    print("Saved score_distributions.png")

# ── 6. Efficiency vs SNR ────────────────────────────────────────────────────────
snr_bin_edges   = np.linspace(4, 30, NUM_BINS + 1)
snr_bin_centers = 0.5 * (snr_bin_edges[:-1] + snr_bin_edges[1:])

fig, ax = plt.subplots(figsize=(9, 6))
for cls in SIGNAL_CLASSES:
    cls_scores = scores_by_class[cls]
    cls_snrs   = data_by_class[cls]["snrs"]
    bin_idx    = np.digitize(cls_snrs, snr_bin_edges) - 1
    frac = [
        np.mean(cls_scores[bin_idx == b] < threshold) if np.any(bin_idx == b) else np.nan
        for b in range(NUM_BINS)
    ]
    ax.plot(snr_bin_centers, frac, marker="o", label=cls)
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Network SNR", fontsize=12)
ax.set_ylabel("Fraction detected", fontsize=12)
ax.set_title(f"Efficiency vs SNR  (threshold = {threshold:.3f})", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "efficiency_vs_snr.png"), dpi=200)
plt.close()
print("Saved efficiency_vs_snr.png")

# ── 7. Efficiency vs hrss ───────────────────────────────────────────────────────
all_sig_hrss     = np.concatenate([data_by_class[c]["hrss"] for c in SIGNAL_CLASSES])
hrss_min         = all_sig_hrss[all_sig_hrss > 0].min()
hrss_max         = all_sig_hrss.max()
hrss_bin_edges   = np.logspace(np.log10(hrss_min), np.log10(hrss_max), NUM_BINS + 1)
hrss_bin_centers = np.sqrt(hrss_bin_edges[:-1] * hrss_bin_edges[1:])

fig, ax = plt.subplots(figsize=(9, 6))
for cls in SIGNAL_CLASSES:
    cls_scores = scores_by_class[cls]
    cls_hrss   = data_by_class[cls]["hrss"]
    bin_idx    = np.digitize(cls_hrss, hrss_bin_edges) - 1
    frac = [
        np.mean(cls_scores[bin_idx == b] < threshold) if np.any(bin_idx == b) else np.nan
        for b in range(NUM_BINS)
    ]
    ax.plot(hrss_bin_centers, frac, marker="o", label=cls)
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xlabel("hrss", fontsize=12)
ax.set_ylabel("Fraction detected", fontsize=12)
ax.set_title(f"Efficiency vs hrss  (threshold = {threshold:.3f})", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "efficiency_vs_hrss.png"), dpi=200)
plt.close()
print("Saved efficiency_vs_hrss.png")

