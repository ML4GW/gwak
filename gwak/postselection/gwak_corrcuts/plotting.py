"""plotting.py — the THREE views per feature (DESIGN §7.2).

For EVERY continuous feature column produce:
  1. overlay            : all-signals (BBH+SG+WNB combined) vs Background [+ Glitch curve]
  2. snrbins_allsignals : single Background distribution + one signal curve per SNR bin
  3. snrbins_<Class>    : one figure per signal class, per-SNR-bin curves vs Background

Conventions (DESIGN §7.2):
  - density-normalized histograms
  - SNR bin edges [3,6,9,12,16,20,25,30], labelled in titles
  - robust 1-99 percentile axis clipping (edr_tile etc. have huge tails); symlog where the
    feature spans many decades and straddles zero / is non-negative with a heavy tail
  - filenames plots/{feature}__{view}.png
  - NaN-gated rows excluded from the histogram; gated fraction reported separately

NOTE: plots are PURELY DESCRIPTIVE. No thresholds are baked in (features stay decoupled from
cut logic). The SNR axis only exists for signal classes (Background/Glitch are SNR=0).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIGNAL_CLASSES = ["BBH", "SineGaussian", "WhiteNoiseBurst"]
BG_CLASS = "Background"
GLITCH_CLASS = "Glitch"

# identity / status / flag columns that are NOT continuous features to histogram
_NON_FEATURE = {
    "event_id", "class_name", "row_index", "snr", "regime", "status", "fail_reason",
    "match_mode",
}
# integer flags that are still informative to plot as distributions
_FLAG_COLS = {
    "bestlag_unconstrained", "single_det_dominated_tile", "low_occupancy_tile",
}

# features that span many decades / have heavy tails -> symlog x where helpful
_HEAVY_TAIL = {
    "edr_tile", "E_tile", "Em_tile", "E_H_tile", "E_L_tile", "L_model_tile",
    "ec_tile", "en_tile", "en_dom_tile", "chi2_tile", "rho_tile",
    "snr_coin_max_tile", "snr_coin_wmean_tile", "snr_ratio_tile", "n_matched",
    "n_coinc_tiles_tile", "n_coinc_clusters_tile", "n_tiles_H", "n_tiles_L",
}


def feature_columns(df):
    cols = []
    for c in df.columns:
        if c in _NON_FEATURE:
            continue
        if df[c].dtype == object:
            continue
        cols.append(c)
    return cols


def _clean(vals):
    """Drop NaN/inf; return finite values + count of finite."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    return v


def _robust_range(arrays, lo=1.0, hi=99.0):
    """1-99 percentile range over the pooled finite values of several arrays."""
    pool = np.concatenate([a for a in arrays if a.size]) if arrays else np.zeros(0)
    if pool.size == 0:
        return None
    a = np.percentile(pool, lo)
    b = np.percentile(pool, hi)
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        # fall back to min/max
        a, b = float(np.min(pool)), float(np.max(pool))
        if a == b:
            b = a + 1.0
    return (a, b)


def _bin_edges(rng, n=60, symlog=False):
    a, b = rng
    if symlog:
        # symmetric-log style edges: linear near 0, log in the tails. Use a signed log spacing
        # only when the range straddles or hugs zero with a big tail; else plain linear is fine.
        # Keep it simple & robust: clip to range and use linear bins (axis set to symlog).
        return np.linspace(a, b, n + 1)
    return np.linspace(a, b, n + 1)


def _hist(ax, vals, edges, label, **kw):
    v = vals[(vals >= edges[0]) & (vals <= edges[-1])]
    if v.size == 0:
        return 0
    ax.hist(v, bins=edges, density=True, histtype="step", linewidth=1.6,
            label="%s (n=%d)" % (label, v.size), **kw)
    return v.size


def _gated_fraction(df, cls, feature):
    sub = df[df["class_name"] == cls]
    if sub.shape[0] == 0:
        return np.nan
    return float(np.mean(~np.isfinite(sub[feature].astype(np.float64))))


def _snr_bin_label(edges, i):
    return "[%g,%g)" % (edges[i], edges[i + 1])


def plot_overlay(df, feature, edges_snr, outdir):
    """View 1: all signals combined vs Background (+ Glitch curve)."""
    sig = _clean(df[df["class_name"].isin(SIGNAL_CLASSES)][feature].values)
    bg = _clean(df[df["class_name"] == BG_CLASS][feature].values)
    gl = _clean(df[df["class_name"] == GLITCH_CLASS][feature].values)
    rng = _robust_range([sig, bg, gl])
    if rng is None:
        return None
    symlog = feature in _HEAVY_TAIL
    edges = _bin_edges(rng, symlog=symlog)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _hist(ax, bg, edges, "Background", color="k")
    _hist(ax, gl, edges, "Glitch", color="0.5", linestyle="--")
    _hist(ax, sig, edges, "All signals", color="C3")
    if symlog:
        ax.set_xscale("symlog")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    gf_sig = float(np.mean(~np.isfinite(
        df[df["class_name"].isin(SIGNAL_CLASSES)][feature].astype(np.float64))))
    gf_bg = _gated_fraction(df, BG_CLASS, feature)
    ax.set_title("%s — overlay (Regime A)\nNaN-gated: signals %.0f%%, bg %.0f%%"
                 % (feature, 100 * gf_sig, 100 * gf_bg), fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "%s__overlay.png" % feature)
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def plot_snrbins_allsignals(df, feature, edges_snr, outdir):
    """View 2: single Background dist + one combined-signal curve per SNR bin."""
    bg = _clean(df[df["class_name"] == BG_CLASS][feature].values)
    sigdf = df[df["class_name"].isin(SIGNAL_CLASSES)]
    # collect per-bin arrays
    bin_arrays = []
    for i in range(len(edges_snr) - 1):
        m = (sigdf["snr"] >= edges_snr[i]) & (sigdf["snr"] < edges_snr[i + 1])
        bin_arrays.append(_clean(sigdf[m][feature].values))
    rng = _robust_range([bg] + bin_arrays)
    if rng is None:
        return None
    symlog = feature in _HEAVY_TAIL
    edges = _bin_edges(rng, symlog=symlog)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _hist(ax, bg, edges, "Background", color="k")
    cmap = plt.cm.viridis
    nb = len(bin_arrays)
    for i, arr in enumerate(bin_arrays):
        if arr.size == 0:
            continue
        _hist(ax, arr, edges, "SNR " + _snr_bin_label(edges_snr, i),
              color=cmap(i / max(nb - 1, 1)))
    if symlog:
        ax.set_xscale("symlog")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.set_title("%s — all signals by SNR (Regime A)\nSNR edges %s"
                 % (feature, list(edges_snr)), fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(outdir, "%s__snrbins_allsignals.png" % feature)
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def plot_snrbins_perclass(df, feature, edges_snr, outdir, cls):
    """View 3: one figure per signal class, per-SNR-bin curves vs Background."""
    bg = _clean(df[df["class_name"] == BG_CLASS][feature].values)
    cdf = df[df["class_name"] == cls]
    bin_arrays = []
    for i in range(len(edges_snr) - 1):
        m = (cdf["snr"] >= edges_snr[i]) & (cdf["snr"] < edges_snr[i + 1])
        bin_arrays.append(_clean(cdf[m][feature].values))
    rng = _robust_range([bg] + bin_arrays)
    if rng is None:
        return None
    symlog = feature in _HEAVY_TAIL
    edges = _bin_edges(rng, symlog=symlog)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _hist(ax, bg, edges, "Background", color="k")
    cmap = plt.cm.plasma
    nb = len(bin_arrays)
    for i, arr in enumerate(bin_arrays):
        if arr.size == 0:
            continue
        _hist(ax, arr, edges, "SNR " + _snr_bin_label(edges_snr, i),
              color=cmap(i / max(nb - 1, 1)))
    if symlog:
        ax.set_xscale("symlog")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    gf = _gated_fraction(df, cls, feature)
    ax.set_title("%s — %s by SNR (Regime A)\nSNR edges %s | %s NaN-gated %.0f%%"
                 % (feature, cls, list(edges_snr), cls, 100 * gf), fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(outdir, "%s__snrbins_%s.png" % (feature, cls))
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def make_all_plots(df, config, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    edges_snr = list(config.snr_bin_edges) if config is not None else \
        [3, 6, 9, 12, 16, 20, 25, 30]
    features = feature_columns(df)
    made = []
    for feat in features:
        try:
            p = plot_overlay(df, feat, edges_snr, outdir)
            if p:
                made.append(p)
            p = plot_snrbins_allsignals(df, feat, edges_snr, outdir)
            if p:
                made.append(p)
            for cls in SIGNAL_CLASSES:
                p = plot_snrbins_perclass(df, feat, edges_snr, outdir, cls)
                if p:
                    made.append(p)
        except Exception as exc:  # never abort the whole batch on one feature
            print("[plot] FAILED %s: %s: %s" % (feat, type(exc).__name__, exc))
    return made, features


def gated_fraction_table(df):
    """Per-class NaN-gated fraction of cc_tile (the headline gate) + per-feature optionally."""
    rows = []
    for cls in df["class_name"].unique():
        sub = df[df["class_name"] == cls]
        rows.append((cls, sub.shape[0],
                     float(np.mean(~np.isfinite(sub["cc_tile"].astype(np.float64)))),
                     float(np.mean(sub["low_occupancy_tile"] == 1))))
    return rows


if __name__ == "__main__":
    import argparse
    from gwak_corrcuts.io import read_feature_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/features.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()
    df, cfg = read_feature_table(args.table)
    made, feats = make_all_plots(df, cfg, args.outdir)
    print("made %d plot files over %d features" % (len(made), len(feats)))
    print("gated-fraction table (class, n, cc_tile NaN frac, low_occupancy frac):")
    for r in gated_fraction_table(df):
        print("  %-16s n=%-6d cc_NaN=%.3f low_occ=%.3f" % r)
