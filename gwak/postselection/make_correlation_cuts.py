"""make_correlation_cuts.py — the GWAK_correlationcuts postselection "correlation cut maker".

SUPERSEDES ml4gw/gwak ``gwak/postselection/cwb_veto_simple.py``.  Same command-line INPUTS and
same CSV OUTPUT *skeleton*, but every metric column is a NATIVE GWAK_correlationcuts tile-space
feature (new names, new formulas) and the headline veto is this project's combined log-LR metric
``C_full`` / ``C_coh`` at fixed background-rejection working points (90% and 99%).

WHY the columns change but the format does not
----------------------------------------------
The reference emitted 8 cWB-name proxies (rho, edr, cc, scc, coh_max, coh_mean, corr_mag_fd,
corr_real_fd).  Those are replaced 1:1 by the native tile features (``cc_tile``, ``ec_tile``,
``en_tile``, ``scc_tile``, ``edr_tile``, ``rho_tile``, the oLIB cross-power / best-lag / coincidence
features, the burst-windowed strain proxies) plus the two fused scores ``C_full`` and ``C_coh``.
The bookkeeping / metadata / veto skeleton is byte-for-byte the same shape:

    segment_idx, success, error,
    t0, length, shift, error_start, error_end, duration, gwak_value,     # metadata (unchanged)
    <NATIVE tile features ...>, C_full, C_coh,                           # metrics (native)
    veto_C_full_90, veto_C_full_99, veto_C_coh_90, veto_C_coh_99,        # requested working points
    vetoed, failed_cuts, n_failed_cuts                                   # veto verdict (unchanged)

INPUT is the SAME as the reference: a GWAK ``error_config.h5`` (the (N,7) ``data`` table) plus the
raw strain ``--data-dir`` of ``background-{t0}-{length}.h5`` files.  See "Two whitening sources"
below for why the native path reaches those raw files through a pre-whitening stage.

Two whitening sources (the native two-interpreter reality)
----------------------------------------------------------
Native tile features (``gwak_corrcuts``) import gwpy and run in /usr/bin/python3 (gwpy 2.1.3, no
torch).  Native whitening is gwak's OWN ``BatchWhitener`` (bit-faithful to the model that produced
error_config.h5) and lives in the gwak deploy venv (py3.11 + torch).  The two cannot share one
interpreter, so this tool runs in the gwpy venv and obtains whitened 1-s kernels one of two ways:

  * ``--kernels results/timeslide_kernels.h5``  — a pre-whitened Stage-1 HDF5 from
    ``recon_timeslide.py`` (``Timeslide_data`` (N,2,4096) + ``Timeslide_ec_row`` → error_config row).
    This is the primary, fully-native path.  RECOMMENDED.

  * ``--data-dir <raw dir>``  — identical input to the reference.  This tool then shells
    ``recon_timeslide.py`` in the gwak venv (``--gwak-python``) to whiten the selected segments
    EXACTLY as gwak did, writing a temporary kernels HDF5, then continues in-process.

Either way the per-segment metadata block (t0/length/shift/error_start/error_end/duration/gwak_value)
is read from ``--error-config`` indexed by the kernel's error_config row — identical to the reference.

RUN (native path):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 make_correlation_cuts.py \
      --error-config /path/to/error_config.h5 \
      --kernels     results/timeslide_kernels_200k.h5 \
      --output      results/correlation_cuts.csv \
      --thresholds  postselection_thresholds_example.json      # optional extra per-feature cuts

RUN (same input as the reference; auto Stage-1 whitening in the gwak venv):
  python3 make_correlation_cuts.py \
      --error-config /path/to/error_config.h5 \
      --data-dir     /home/hongyin.chen/Data/O4_MDC_short-0/HL \
      --output       results/correlation_cuts.csv --first-n 20000
"""
import os

# MANDATORY before numpy import (RLIMIT_NPROC segfault / OpenBLAS thread storm on py3.6 otherwise).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------------------------
# Defaults (documented in recon_timeslide.py / DESIGN_timeslide.md)
# --------------------------------------------------------------------------------------------
DEFAULT_MODEL = "results/combined_model.json"
DEFAULT_CALIB = "results/calibration_hrss.json"
DEFAULT_GWAK_DEPLOY = "/home/hongyin.chen/anti_gravity/gwak/gwak/deploy"
DEFAULT_GWAK_PYTHON = "/home/hongyin.chen/anti_gravity/gwak/gwak/deploy/.venv/bin/python"

FS = 4096.0

# error_config (N,7) column layout (see recon_timeslide.py §C_*).
C_T0, C_LEN, C_SHIFT, C_ESTART, C_EEND, C_DUR, C_GWAK = range(7)
METADATA_KEYS = ["t0", "length", "shift", "error_start", "error_end", "duration", "gwak_value"]

# Native tile features, in output-CSV order (matches the emitted feature table header exactly).
FEATURE_ORDER = [
    "E_tile", "Em_tile", "E_H_tile", "E_L_tile", "L_model_tile",
    "ec_tile", "en_tile", "cc_tile", "chi2_tile", "rho_tile", "scc_tile", "edr_tile",
    "cc_dom_tile", "en_dom_tile", "cc_offdiag_tile",
    "xcorr_signed_tile", "xcorr_mag_tile", "xcorr_energy_proxy",
    "best_lag_tile", "xcorr_bestlag_tile", "xcorr_zerolag_tile", "bestlag_unconstrained",
    "dominance_tile", "participation_H_tile", "participation_L_tile",
    "n_tiles_H", "n_tiles_L", "n_matched",
    "single_det_dominated_tile", "low_occupancy_tile",
    "n_coinc_tiles_tile", "n_coinc_clusters_tile", "f0_agreement_tile", "Q_agreement_tile",
    "dt_consistency_tile", "coinc_energy_frac_tile", "min_coinc_frac_tile",
    "snr_coin_max_tile", "snr_coin_wmean_tile", "snr_ratio_tile",
    "strain_xcorr_win_proxy", "strain_abslag_ms_proxy",
]

# The two fused scores + the four requested working-point veto flags.
SCORE_COLS = ["C_full", "C_coh"]
WP_COLS = ["veto_C_full_90", "veto_C_full_99", "veto_C_coh_90", "veto_C_coh_99"]


# ============================================================================================
# Combined-metric scorer (bit-faithful reimplementation of combined_metric.apply_preproc+score)
# ============================================================================================
class ComboScorer:
    """Score C_full / C_coh from a persisted combined_model.json.

    Reproduces analysis/combined_metric.py exactly: per feature, inf->NaN, NaN->train sentinel,
    clip to +/-train-clip, z-score by train (mean,std), then w . z + b.  Neyman-Pearson-optimal
    log-LR up to a constant, so a threshold at a fixed background quantile IS a fixed-rejection cut.
    """

    def __init__(self, model_json, trained_vs):
        with open(model_json) as fh:
            self.M = json.load(fh)
        self.trained_vs = trained_vs
        self._compiled = {}
        for fs in SCORE_COLS:
            key = "%s__%s" % (fs, trained_vs)
            if key not in self.M["models"]:
                raise KeyError("model %r not in %s (have %s)"
                               % (key, model_json, list(self.M["models"])))
            m = self.M["models"][key]
            feats = m["feats"]
            pp = m["preproc"]
            self._compiled[fs] = dict(
                feats=feats,
                w=np.asarray([m["weights"][f] for f in feats], dtype=np.float64),
                b=float(m["bias"]),
                sentinel=np.asarray(pp["sentinel"], dtype=np.float64),
                clip=np.asarray(pp["clip"], dtype=np.float64),
                mean=np.asarray(pp["mean"], dtype=np.float64),
                std=np.asarray(pp["std"], dtype=np.float64),
                thr=m["thr_for_rej"],
            )

    def score(self, row, fs):
        c = self._compiled[fs]
        v = np.asarray([row.get(f, np.nan) for f in c["feats"]], dtype=np.float64)
        v = np.where(np.isfinite(v), v, c["sentinel"])
        v = np.clip(v, -c["clip"], c["clip"])
        z = (v - c["mean"]) / c["std"]
        return float(z @ c["w"] + c["b"])

    def threshold(self, fs, rej):
        """The trained fixed-rejection threshold: events with score < thr are vetoed
        (that removes fraction `rej` of the reject population by construction)."""
        return float(self._compiled[fs]["thr"][str(rej)])


# ============================================================================================
# Burst-windowed strain coherence proxies (ported verbatim from run_extraction_timeslide.py)
# ============================================================================================
def strain_proxies(xH, xL, win_half=256, max_lag=61):
    """Returns (peak |xcorr| in a burst window, |lag| ms).  Tile-pipeline-independent; carries
    real weight in the combined model (strain_xcorr_win_proxy)."""
    from scipy.signal import hilbert
    xH = np.asarray(xH, dtype=np.float64)
    xL = np.asarray(xL, dtype=np.float64)
    eH = np.abs(hilbert(xH - xH.mean()))
    eL = np.abs(hilbert(xL - xL.mean()))
    env = eH * eH + eL * eL
    k = 33
    sm = np.convolve(env, np.ones(k) / k, mode="same")
    c = int(np.argmax(sm))
    lo = max(0, c - win_half)
    hi = min(len(xH), c + win_half)
    a = xH[lo:hi]
    b = xL[lo:hi]
    a = a - a.mean()
    b = b - b.mean()
    na = np.sqrt((a * a).sum())
    nb = np.sqrt((b * b).sum())
    if na == 0 or nb == 0:
        return np.nan, np.nan
    full = np.correlate(a, b, mode="full") / (na * nb)
    lags = np.arange(-(len(b) - 1), len(a))
    m = np.abs(lags) <= max_lag
    seg = full[m]
    seglags = lags[m]
    i = int(np.argmax(np.abs(seg)))
    return float(np.abs(seg[i])), float(abs(seglags[i]) / FS * 1000.0)


# ============================================================================================
# error_config metadata
# ============================================================================================
def load_error_config(path):
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f["data"][:], dtype=np.float64)


def metadata_for(ec, seg_idx):
    """The reference's metadata block, read from error_config row `seg_idx`."""
    r = ec[seg_idx]
    return {
        "t0": int(r[C_T0]),
        "length": int(r[C_LEN]),
        "shift": int(r[C_SHIFT]),
        "error_start": float(r[C_ESTART]),
        "error_end": float(r[C_EEND]),
        "duration": float(r[C_DUR]),
        "gwak_value": float(r[C_GWAK]),
    }


# ============================================================================================
# whitened-kernel sources
# ============================================================================================
def load_kernels(path):
    """Load a Stage-1 (recon_timeslide.py) HDF5.  Returns (data (n,2,Ns), ec_row (n,))."""
    import h5py
    with h5py.File(path, "r") as f:
        data = np.asarray(f["Timeslide_data"][:], dtype=np.float32)
        ec_row = np.asarray(f["Timeslide_ec_row"][:], dtype=np.int64)
        attrs = dict(f.attrs)
    return data, ec_row, attrs


def whiten_from_raw(error_config, data_dir, out_h5, first_n, device, gwak_python, gwak_deploy):
    """Same input as the reference (raw error_config + data-dir) → whiten the selected segments
    EXACTLY as gwak did by shelling recon_timeslide.py in the gwak deploy venv.  strategy='all'
    keeps error_config/file order so segment_idx is natural."""
    here = os.path.dirname(os.path.abspath(__file__))
    recon = os.path.join(here, "recon_timeslide.py")
    cmd = [gwak_python, recon,
           "--error-config", error_config,
           "--data-dir", data_dir,
           "--out", out_h5,
           "--strategy", "all",
           "--gwak-deploy", gwak_deploy]
    if first_n is not None:
        cmd += ["--max-events", str(first_n)]
    if device in ("cuda", "cpu"):
        cmd += ["--device", device]
    print("[stage1] whitening via gwak venv:\n  %s" % " ".join(cmd), flush=True)
    if not os.path.exists(gwak_python):
        raise SystemExit(
            "gwak venv python not found: %s\n"
            "Native whitening (BatchWhitener) lives in the gwak deploy venv (py3.11+torch), which "
            "cannot share this gwpy interpreter. Either pass --gwak-python <path> or run Stage 1 "
            "yourself and pass --kernels:\n"
            "  PYTHONPATH=%s %s/.venv/bin/python recon_timeslide.py "
            "--error-config ... --data-dir ... --out kernels.h5 --strategy all"
            % (gwak_python, gwak_deploy, gwak_deploy))
    subprocess.run(cmd, check=True)
    return out_h5


# ============================================================================================
# per-segment processing
# ============================================================================================
_G = {}


def _worker_init(cfg_dict, model_json, trained_vs):
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    from gwak_corrcuts.config import Config
    from gwak_corrcuts.tiles import build_grid
    cfg = Config.from_dict(cfg_dict)
    _G["cfg"] = cfg
    _G["grid"] = build_grid(cfg)
    _G["scorer"] = ComboScorer(model_json, trained_vs)


def _process_one(task):
    """task = (seg_idx, xH, xL, meta_dict). Returns the fully-assembled feature+score+meta row
    (without the veto verdict, which is applied centrally so thresholds/working-points stay in
    one place)."""
    from gwak_corrcuts.pipeline import process_event
    seg_idx, xH, xL, meta = task
    cfg = _G["cfg"]
    grid = _G["grid"]
    scorer = _G["scorer"]

    out = {"segment_idx": int(seg_idx), "success": True, "error": None}
    out.update(meta)
    try:
        row = process_event(xH, xL, 0.0, "Timeslide", int(seg_idx), grid, cfg)
        status = row.get("status", "ok")
        # native status -> reference success/error mapping
        if status != "ok":
            out["success"] = False
            out["error"] = row.get("fail_reason") or ("status=%s" % status)
        try:
            sx, slag = strain_proxies(xH, xL)
        except Exception:
            sx, slag = np.nan, np.nan
        row["strain_xcorr_win_proxy"] = sx
        row["strain_abslag_ms_proxy"] = slag
        for k in FEATURE_ORDER:
            out[k] = row.get(k, np.nan)
        out["C_full"] = scorer.score(row, "C_full")
        out["C_coh"] = scorer.score(row, "C_coh")
    except Exception as exc:  # never abort the run (mirrors the reference)
        out["success"] = False
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        for k in FEATURE_ORDER:
            out[k] = np.nan
        out["C_full"] = np.nan
        out["C_coh"] = np.nan
    return out


# ============================================================================================
# veto verdict
# ============================================================================================
def apply_vetoes(row, thresholds, scorer, headline_metric, headline_rej):
    """Reference-compatible veto: per-feature {min,max} from the thresholds JSON PLUS the combined
    working points.  Emits the four requested flags (C_full/C_coh @ 90%/99% rejection) always, and
    an aggregate `vetoed`/`failed_cuts`/`n_failed_cuts` for the chosen headline operating point."""
    vetoed = False
    failed = []

    # (1) optional per-feature thresholds — identical semantics to cwb_veto_simple.apply_vetoes
    for feat, th in (thresholds or {}).items():
        if feat.startswith("_") or feat not in row or not isinstance(th, dict):
            continue  # skip comments / non-feature keys
        val = row[feat]
        lo = th.get("min", None)
        hi = th.get("max", None)
        if lo is not None and (not np.isfinite(val) or val < lo):
            vetoed = True
            failed.append("%s < %s" % (feat, lo))
        if hi is not None and (not np.isfinite(val) or val > hi):
            vetoed = True
            failed.append("%s > %s" % (feat, hi))

    # (2) combined-metric working points (the headline correlation cut)
    wp = {}
    for fs in SCORE_COLS:
        sc = row.get(fs, np.nan)
        for rej in (0.90, 0.99):
            thr = scorer.threshold(fs, rej)
            fail = (not np.isfinite(sc)) or (sc < thr)
            wp["veto_%s_%d" % (fs, int(rej * 100))] = bool(fail)

    # (3) aggregate verdict at the chosen headline working point
    hsc = row.get(headline_metric, np.nan)
    hthr = scorer.threshold(headline_metric, headline_rej)
    if (not np.isfinite(hsc)) or (hsc < hthr):
        vetoed = True
        failed.append("%s < %.6f (%d%% rej)" % (headline_metric, hthr, int(headline_rej * 100)))

    row.update(wp)
    row["vetoed"] = bool(vetoed)
    row["failed_cuts"] = ";".join(failed)
    row["n_failed_cuts"] = len(failed)
    return row


# ============================================================================================
# main
# ============================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- reference-compatible arguments ---
    ap.add_argument("--error-config", required=True, help="GWAK error_config.h5 (metadata source)")
    ap.add_argument("--data-dir", default=None,
                    help="raw background-{t0}-{len}.h5 dir (same input as the reference); "
                         "auto-whitened via the gwak venv when --kernels is not given")
    ap.add_argument("--output", required=True, help="output CSV")
    ap.add_argument("--thresholds", default=None,
                    help="optional JSON of {feature: {min, max}} extra per-feature cuts "
                         "(same shape as the reference's thresholds file)")
    ap.add_argument("--first-n", type=int, default=None, help="process only the first N segments")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="device for Stage-1 whitening (raw --data-dir mode)")
    # --- native extras ---
    ap.add_argument("--kernels", default=None,
                    help="pre-whitened Stage-1 HDF5 from recon_timeslide.py (Timeslide_data/_ec_row)")
    ap.add_argument("--model-json", default=DEFAULT_MODEL, help="combined_model.json (C_full/C_coh)")
    ap.add_argument("--calib", default=DEFAULT_CALIB,
                    help="calibration_*.json (E_thr + kappa_noise) so tiles match the trained model")
    ap.add_argument("--trained-vs", default="R2_timeslide",
                    choices=["R2_timeslide", "R1_bg_glitch"],
                    help="which reject population's combined model to veto with")
    ap.add_argument("--veto-metric", default="C_full", choices=["C_full", "C_coh"],
                    help="combined metric used for the aggregate `vetoed` verdict")
    ap.add_argument("--veto-rej", type=float, default=0.90, choices=[0.90, 0.99],
                    help="background-rejection working point for the aggregate `vetoed` verdict")
    ap.add_argument("--workers", type=int, default=1, help=">1 enables a spawn Pool")
    ap.add_argument("--gwak-python", default=DEFAULT_GWAK_PYTHON,
                    help="python of the gwak deploy venv (raw --data-dir mode)")
    ap.add_argument("--gwak-deploy", default=DEFAULT_GWAK_DEPLOY,
                    help="gwak deploy dir (raw --data-dir mode)")
    args = ap.parse_args(argv)

    if args.kernels is None and args.data_dir is None:
        ap.error("provide either --kernels (pre-whitened) or --data-dir (raw, auto-whitened)")

    # --- config with native calibration (tiles MUST match the trained model) -----------------
    from gwak_corrcuts.config import Config
    cfg = Config(regime="A")
    calib = None
    if os.path.exists(args.calib):
        with open(args.calib) as fh:
            calib = json.load(fh)
        cfg.e_thr_calibrated = calib["e_thr"]
        cfg.kappa_noise = calib["kappa_noise"]
        print("[calib] e_thr=%.4f kappa_noise=%.4f (%s)"
              % (calib["e_thr"], calib["kappa_noise"], args.calib), flush=True)
    else:
        print("[calib] WARNING: %s not found — using uncalibrated tile selection; scores will not "
              "match the trained model." % args.calib, flush=True)

    scorer = ComboScorer(args.model_json, args.trained_vs)
    ec = load_error_config(args.error_config)
    print("[load] error_config rows=%d  model=%s (trained_vs=%s)"
          % (ec.shape[0], args.model_json, args.trained_vs), flush=True)

    thresholds = None
    if args.thresholds and os.path.exists(args.thresholds):
        with open(args.thresholds) as fh:
            thresholds = json.load(fh)
        # tolerate a nested {"features": {...}} layout as well as the flat reference layout
        if "features" in thresholds and isinstance(thresholds["features"], dict):
            thresholds = thresholds["features"]
        n_cuts = sum(1 for k, v in thresholds.items()
                     if not k.startswith("_") and isinstance(v, dict))
        print("[thresholds] %d extra per-feature cuts from %s"
              % (n_cuts, args.thresholds), flush=True)

    # --- obtain whitened kernels -------------------------------------------------------------
    tmp_h5 = None
    kernels_path = args.kernels
    if kernels_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="corrcuts_kernels_", suffix=".h5", delete=False)
        tmp.close()
        tmp_h5 = tmp.name
        kernels_path = whiten_from_raw(args.error_config, args.data_dir, tmp_h5, args.first_n,
                                       args.device, args.gwak_python, args.gwak_deploy)

    data, ec_row, kattrs = load_kernels(kernels_path)
    print("[kernels] %s  n=%d  shape=%s" % (kernels_path, data.shape[0], data.shape), flush=True)

    if args.first_n is not None and args.first_n < data.shape[0]:
        data = data[:args.first_n]
        ec_row = ec_row[:args.first_n]
        print("[kernels] --first-n -> processing %d segments" % data.shape[0], flush=True)

    # --- build tasks -------------------------------------------------------------------------
    tasks = []
    for i in range(data.shape[0]):
        seg = int(ec_row[i])
        tasks.append((seg, np.asarray(data[i, 0], dtype=np.float32),
                      np.asarray(data[i, 1], dtype=np.float32), metadata_for(ec, seg)))

    # --- process -----------------------------------------------------------------------------
    t0 = time.time()
    rows = []
    if args.workers and args.workers > 1:
        cfg_dict = cfg.to_dict()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_worker_init,
                      initargs=(cfg_dict, args.model_json, args.trained_vs)) as pool:
            for k, row in enumerate(pool.imap(_process_one, tasks, chunksize=8)):
                rows.append(row)
                if (k + 1) % 5000 == 0:
                    el = time.time() - t0
                    print("[cut] %d/%d  %.1fs  %.4f s/ev"
                          % (k + 1, len(tasks), el, el / (k + 1)), flush=True)
    else:
        _worker_init(cfg.to_dict(), args.model_json, args.trained_vs)
        for k, task in enumerate(tasks):
            rows.append(_process_one(task))
            if (k + 1) % 2000 == 0:
                el = time.time() - t0
                print("[cut] %d/%d  %.1fs  %.4f s/ev"
                      % (k + 1, len(tasks), el, el / (k + 1)), flush=True)
    wall = time.time() - t0

    # --- veto verdict ------------------------------------------------------------------------
    for row in rows:
        apply_vetoes(row, thresholds, scorer, args.veto_metric, args.veto_rej)

    # --- write CSV in the reference skeleton -------------------------------------------------
    columns = (["segment_idx", "success", "error"] + METADATA_KEYS
               + FEATURE_ORDER + SCORE_COLS + WP_COLS
               + ["vetoed", "failed_cuts", "n_failed_cuts"])
    import pandas as pd
    df = pd.DataFrame(rows).reindex(columns=columns)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)

    # native provenance sidecars (config + run metadata)
    with open(args.output + ".config.json", "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    n_success = int(df["success"].sum())
    n_vetoed = int(df.loc[df["success"], "vetoed"].sum()) if n_success else 0
    meta = {
        "error_config": args.error_config, "kernels": kernels_path, "data_dir": args.data_dir,
        "model_json": args.model_json, "trained_vs": args.trained_vs,
        "veto_metric": args.veto_metric, "veto_rej": args.veto_rej,
        "calibration": calib, "kernels_attrs": {k: _js(v) for k, v in kattrs.items()},
        "n_segments": int(df.shape[0]), "n_success": n_success, "n_vetoed": n_vetoed,
        "working_point_thresholds": {
            fs: {r: scorer.threshold(fs, r) for r in (0.90, 0.99)} for fs in SCORE_COLS},
        "wall_s": wall, "workers": args.workers,
    }
    with open(args.output + ".runmeta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    # --- summary (mirrors the reference) -----------------------------------------------------
    print("\n[cut] DONE %d segments in %.1fs (%.4f s/ev)"
          % (len(rows), wall, wall / max(len(rows), 1)), flush=True)
    print("Successfully processed: %d/%d" % (n_success, len(df)))
    if n_success:
        print("Vetoed (aggregate, %s@%d%% rej): %d/%d  Passed: %d/%d"
              % (args.veto_metric, int(args.veto_rej * 100), n_vetoed, n_success,
                 n_success - n_vetoed, n_success))
        for c in WP_COLS:
            print("  %-16s vetoed %d/%d" % (c, int(df.loc[df["success"], c].sum()), n_success))
    print("Results saved to: %s" % args.output)

    if tmp_h5 is not None:
        try:
            os.unlink(tmp_h5)
        except OSError:
            pass
    return 0


def _js(v):
    """Make h5 attr values JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


if __name__ == "__main__":
    sys.exit(main())
