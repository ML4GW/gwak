#!/usr/bin/env python3
"""
Run one-month background evaluation using either:
  - Combined JIT model (embedding + NF), when only --model-path is given
  - Embedding model + Isolation Forest, when --if-model is also given

Each H5 file in inference_result/ is expected to contain:
    data : (N, 2, 4096) float32 array of whitened strain chunks

Usage (NF):
    python evaluate_one_month.py \
        --model-path output/model/combination/model_JIT.pt \
        --inference-dir output/infer/.../inference_result \
        --output-dir output/model/evaluation/

Usage (IF):
    python evaluate_one_month.py \
        --model-path output/model/model_JIT.pt \
        --if-model output/model/isolation_forest.joblib \
        --means output/model/means.npy \
        --stds  output/model/stds.npy \
        --inference-dir output/infer/.../inference_result \
        --output-dir output/model_IF/evaluation/
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def load_chunks(inference_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(inference_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in:\n  {inference_dir}")
    print(f"Found {len(files)} H5 files.")
    return files


def frequency_cos_similarity(batch: torch.Tensor) -> np.ndarray:
    H = torch.fft.rfft(batch[:, 0, :], dim=-1)
    L = torch.fft.rfft(batch[:, 1, :], dim=-1)
    rho = torch.real(
        torch.sum(H * torch.conj(L), dim=-1) /
        (torch.linalg.norm(H, dim=-1) * torch.linalg.norm(L, dim=-1) + 1e-8)
    )
    return rho.cpu().numpy().reshape(-1, 1)


def run_nf(model, files, device, batch_size) -> np.ndarray:
    all_scores = []
    for fpath in tqdm(files, desc="Inferring"):
        with h5py.File(fpath, "r") as f:
            data = f["data"][:].astype(np.float32)
        valid = np.isfinite(data).all(axis=(1, 2))
        if not valid.all():
            print(f"  [warn] {os.path.basename(fpath)}: skipping {(~valid).sum()} NaN/Inf chunks")
        data = data[valid]
        if len(data) == 0:
            continue
        for start in range(0, len(data), batch_size):
            x = torch.tensor(data[start:start + batch_size]).to(device)
            with torch.no_grad():
                out = model(x)
            all_scores.append(out.cpu().numpy().reshape(-1))
    scores = np.concatenate(all_scores).astype(np.float32)
    n_nan = np.sum(~np.isfinite(scores))
    if n_nan:
        print(f"  [warn] dropping {n_nan} NaN/Inf scores")
        scores = scores[np.isfinite(scores)]
    return scores


def run_if(embed_model, clf, means, stds, files, device, batch_size) -> np.ndarray:
    all_scores = []
    for fpath in tqdm(files, desc="Inferring"):
        with h5py.File(fpath, "r") as f:
            data = f["data"][:].astype(np.float32)
        valid = np.isfinite(data).all(axis=(1, 2))
        if not valid.all():
            print(f"  [warn] {os.path.basename(fpath)}: skipping {(~valid).sum()} NaN/Inf chunks")
        data = data[valid]
        if len(data) == 0:
            continue
        for start in range(0, len(data), batch_size):
            chunk = torch.tensor(data[start:start + batch_size]).to(device)
            with torch.no_grad():
                embeddings = embed_model(chunk).cpu().numpy()
            xcorr = frequency_cos_similarity(chunk)
            embeddings_norm = (embeddings - means) / (stds + 1e-8)
            feats = np.concatenate([embeddings_norm, xcorr], axis=1)
            all_scores.append(clf.score_samples(feats).astype(np.float32))
    return np.concatenate(all_scores)


def tophat_smooth(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(scores, kernel, mode='same')


def veto_long_glitch_segments(scores: np.ndarray, veto_duration: int) -> np.ndarray:
    flag_threshold = np.percentile(scores, 1)
    flagged = scores < flag_threshold
    keep = np.ones(len(scores), dtype=bool)
    i = 0
    while i < len(scores):
        if flagged[i]:
            j = i
            while j < len(scores) and flagged[j]:
                j += 1
            if (j - i) > veto_duration:
                keep[i:j] = False
            i = j
        else:
            i += 1
    print(f"10s veto: removed {(~keep).sum():,} chunks")
    return scores[keep]


def plot_histogram(scores: np.ndarray, output_dir: str, mode: str) -> None:
    color = "darkorange" if mode == "if" else "steelblue"
    label = "IF Anomaly Score (lower = more anomalous)" if mode == "if" else "GWAK Score"
    out = os.path.join(output_dir, "score_histogram.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=300, color=color, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — one month\nn={len(scores):,}  min={scores.min():.3f}  max={scores.max():.3f}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Histogram saved to: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path",    required=True,
                    help="Combined JIT model (NF mode) or embedding JIT model (IF mode)")
    ap.add_argument("--if-model",      default=None,
                    help="Path to isolation_forest.joblib — triggers IF mode")
    ap.add_argument("--means",         default=None, help="means.npy for IF normalization")
    ap.add_argument("--stds",          default=None, help="stds.npy for IF normalization")
    ap.add_argument("--inference-dir", required=True)
    ap.add_argument("--batch-size",    type=int, default=256)
    ap.add_argument("--output-dir",    default="./evaluation_output")
    ap.add_argument("--smooth-window", type=int, default=1)
    ap.add_argument("--veto-duration", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    files = load_chunks(args.inference_dir)

    if args.if_model:
        import joblib
        print(f"Mode: Isolation Forest")
        print(f"Loading embedding model: {args.model_path}")
        embed_model = torch.jit.load(args.model_path, map_location=device)
        embed_model.eval()
        print(f"Loading Isolation Forest: {args.if_model}")
        clf = joblib.load(args.if_model)
        means = np.load(args.means).astype(np.float32)
        stds  = np.load(args.stds).astype(np.float32)
        scores = run_if(embed_model, clf, means, stds, files, device, args.batch_size)
        mode = "if"
    else:
        print(f"Mode: Normalizing Flow")
        print(f"Loading model: {args.model_path}")
        model = torch.jit.load(args.model_path, map_location=device)
        model.eval()
        scores = run_nf(model, files, device, args.batch_size)
        mode = "nf"

    scores = scores[np.isfinite(scores)]

    if args.smooth_window > 1:
        print(f"Applying top-hat smoothing: window={args.smooth_window}")
        scores = tophat_smooth(scores, args.smooth_window)

    if args.veto_duration > 0:
        print(f"Applying {args.veto_duration}s long-segment veto")
        scores = veto_long_glitch_segments(scores, args.veto_duration)

    scores_path = os.path.join(args.output_dir, "scores.npy")
    np.save(scores_path, scores)
    print(f"Scores saved to: {scores_path}")

    plot_histogram(scores, args.output_dir, mode)


if __name__ == "__main__":
    main()
