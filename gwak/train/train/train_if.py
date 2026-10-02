#!/usr/bin/env python3
"""
Train an Isolation Forest on precomputed background embeddings + xcorr.

Usage:
    python train/train_isolation_forest.py \
        --embeddings output/ResNet_HL/embeddings.npy \
        --labels     output/ResNet_HL/labels.npy \
        --correlations output/ResNet_HL/correlations.npy \
        --means      output/ResNet_HL/means.npy \
        --stds       output/ResNet_HL/stds.npy \
        --output     output/ResNet_HL/isolation_forest.joblib
"""
import argparse
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings",   required=True)
parser.add_argument("--labels",       required=True)
parser.add_argument("--correlations", required=True)
parser.add_argument("--means",        required=True)
parser.add_argument("--stds",         required=True)
parser.add_argument("--output",       required=True)
parser.add_argument("--n-estimators", type=int, default=1000)
parser.add_argument("--contamination", type=float, default=0.01)
args = parser.parse_args()

embeddings   = np.load(args.embeddings).astype(np.float32)
labels       = np.load(args.labels).astype(np.int32)
correlations = np.load(args.correlations).astype(np.float32)
means        = np.load(args.means).astype(np.float32)
stds         = np.load(args.stds).astype(np.float32)

# normalize embeddings
embeddings_norm = (embeddings - means) / (stds + 1e-8)

# concatenate xcorr as extra feature
if correlations.ndim == 1:
    correlations = correlations.reshape(-1, 1)
feats = np.concatenate([embeddings_norm, correlations], axis=1)

# Background and Glitch are always the two largest label values
unique_labels = np.unique(labels)
print(f"Unique labels found: {unique_labels}")
bg_labels = sorted(unique_labels)[-2:]
bg_mask = np.isin(labels, bg_labels)
feats_bg = feats[bg_mask]
print(f"Background labels: {bg_labels} — training on {feats_bg.shape[0]:,} samples, {feats_bg.shape[1]}d features")

clf = IsolationForest(
    n_estimators=args.n_estimators,
    contamination=args.contamination,
    random_state=42,
    n_jobs=-1,
)
clf.fit(feats_bg)

joblib.dump(clf, args.output)
print(f"Saved Isolation Forest to {args.output}")

