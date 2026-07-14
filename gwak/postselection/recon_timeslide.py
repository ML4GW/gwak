#!/usr/bin/env python
"""recon_timeslide.py — Stage 1 of the timeslide-background mode.

Ingest a GWAK ``error_config.h5`` and RECREATE each timeslide-background event EXACTLY as
ml4gw/gwak produced it, then whiten it with gwak's OWN ``BatchWhitener`` (which wraps ml4gw
``SpectralDensity`` + ``Whiten``).  Output is an HDF5 in the project dataset schema
(``Timeslide_data`` (N,2,4096) / ``Timeslide_snrs`` (N,)=0 + provenance) that the existing
``gwak_corrcuts`` feature pipeline consumes unchanged.  See ``DESIGN_timeslide.md``.

WHY a separate script / interpreter: whitening must use the SAME functions gwak used, which live
in the gwak deploy venv (py3.11 + torch + vendored ml4gw).  The downstream feature extraction
runs in /usr/bin/python3 (gwpy 2.1.3).  The two stages communicate only through the HDF5 file.

RUN (in the gwak deploy venv):
  GWAK=/home/hongyin.chen/anti_gravity/gwak/gwak/deploy
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=$GWAK $GWAK/.venv/bin/python recon_timeslide.py \
      --out results/timeslide_kernels.h5 --max-events 50000 --strategy significance

For the vertical slice use e.g. ``--limit 400`` (cap on rows actually whitened).
"""
import os

# MANDATORY before numpy/torch import (RLIMIT_NPROC segfault / OpenBLAS thread storm otherwise).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import re
import sys
import time

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Deploy run configuration (verified against
#   output/.../torch_rbw_zp_resnet_do6_dcs128_epoch25_HL/config.yaml
#   deploy/deploy/config/{export,analysis,infer_condor}.yaml).
# These are the numbers the GWAK model that produced error_config.h5 was run with.
# ---------------------------------------------------------------------------
SR = 4096                 # sample_rate (Hz)
KERNEL_LENGTH = 1.0       # s  -> kernel_size = 4096 samples
PSD_LENGTH = 64           # s  (PSD estimation window)
FDURATION = 2             # s  (whitener filter length; crops fduration/2 each edge)
FFTLENGTH = 2             # s  (PSD FFT length)
HIGHPASS = 30             # Hz (bandpass low edge AND whitener highpass)
INFER_RATE = 4            # Hz (inference_sampling_rate)

# Alignment: H1_time (an error_config anomaly GPS) is the CENTER of the scored 1-s kernel.
#   window = [T_H1 - PRE, T_H1 + POST], total = PSD_LENGTH + FDURATION + KERNEL_LENGTH = 67 s.
PRE = PSD_LENGTH + FDURATION / 2.0 + KERNEL_LENGTH / 2.0   # 65.5 s  (start before T_H1)
POST = FDURATION / 2.0 + KERNEL_LENGTH / 2.0               # 1.5 s   (end after T_H1)
WINDOW_S = PSD_LENGTH + FDURATION + KERNEL_LENGTH          # 67 s
WINDOW_N = int(round(WINDOW_S * SR))                       # 274432 samples
KERNEL_N = int(round(KERNEL_LENGTH * SR))                  # 4096

# error_config (N,7) column indices (see DESIGN_timeslide.md §1).
C_T0, C_LEN, C_SHIFT, C_ESTART, C_EEND, C_DUR, C_GWAK = range(7)

DEFAULT_EC = ("/home/hongyin.chen/anti_gravity/gwak/gwak/output/plots/"
              "torch_rbw_zp_resnet_do6_dcs128_epoch25_NF_from_file_conditioning_HL/"
              "one_year/error_config.h5")
DEFAULT_DATADIR = "/home/hongyin.chen/Data/O4_MDC_short-0/HL"
DEFAULT_GWAK_DEPLOY = "/home/hongyin.chen/anti_gravity/gwak/gwak/deploy"


def build_gps_index(data_dir):
    """Map each raw background file to its GPS interval.  Returns sorted arrays + file list."""
    starts, stops, files = [], [], []
    for fn in glob.glob(os.path.join(data_dir, "*.h5")):
        m = re.search(r"background-(\d+)-(\d+)\.h5", os.path.basename(fn))
        if not m:
            continue
        a = int(m.group(1)); length = int(m.group(2))
        starts.append(a); stops.append(a + length); files.append(fn)
    if not starts:
        raise SystemExit("No background-*.h5 files in %s" % data_dir)
    order = np.argsort(starts)
    starts = np.asarray(starts)[order]
    stops = np.asarray(stops)[order]
    files = [files[i] for i in order]
    return starts, stops, files


def select_rows(d, starts, stops, strategy, max_events, score_thr, seed):
    """Return (idx_sel, file_idx_sel, info) — indices of valid+selected rows, the covering
    file index per row, and a stats dict.  Validity = covering file found AND the 67-s window
    (incl. the L1 +shift) lies inside that file's [GPS_start, GPS_stop]."""
    shift = d[:, C_SHIFT].astype(np.int64)
    T = (d[:, C_ESTART] + d[:, C_EEND]) / 2.0          # H1 kernel center (GPS)
    win_s = T - PRE
    win_e = T + POST
    # covering file: largest start <= T, and T < its stop
    j = np.searchsorted(starts, T, side="right") - 1
    j_clip = np.clip(j, 0, len(starts) - 1)
    has_file = (j >= 0) & (T < stops[j_clip])
    a = starts[j_clip].astype(np.float64)
    b = stops[j_clip].astype(np.float64)
    in_bounds = has_file & (win_s >= a) & (win_e + shift <= b)

    valid = np.where(in_bounds)[0]
    n_total = d.shape[0]
    n_nofile = int((~has_file).sum())
    n_oob = int((has_file & ~in_bounds).sum())

    g = d[valid, C_GWAK]
    if score_thr is not None:
        keep = g < score_thr
        valid = valid[keep]; g = g[keep]
    if strategy == "significance":
        order = np.argsort(g)                          # most negative (most significant) first
        valid = valid[order]
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        rng.shuffle(valid)
    elif strategy == "all":
        valid = valid[np.argsort(valid)]               # file/time order
    else:
        raise SystemExit("unknown --strategy %r" % strategy)
    if max_events is not None and max_events < valid.size:
        valid = valid[:max_events]

    file_idx = np.clip(np.searchsorted(starts, T[valid], side="right") - 1, 0, len(starts) - 1)
    info = dict(n_total=n_total, n_nofile=n_nofile, n_oob=n_oob, n_valid_all=int(in_bounds.sum()),
                n_selected=int(valid.size))
    return valid, file_idx, T, shift, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--error-config", default=DEFAULT_EC)
    ap.add_argument("--data-dir", default=DEFAULT_DATADIR,
                    help="dir of raw background-{gps}-{len}.h5 (short-0, canonical)")
    ap.add_argument("--out", default="results/timeslide_kernels.h5")
    ap.add_argument("--strategy", default="significance",
                    choices=["significance", "random", "all"])
    ap.add_argument("--max-events", type=int, default=50000)
    ap.add_argument("--score-thr", type=float, default=None,
                    help="optional: only rows with gwak_value < this")
    ap.add_argument("--limit", type=int, default=None,
                    help="hard cap on rows actually whitened (slice/debug)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--gwak-deploy", default=DEFAULT_GWAK_DEPLOY)
    ap.add_argument("--progress", type=int, default=2000)
    args = ap.parse_args()

    if args.gwak_deploy not in sys.path:
        sys.path.insert(0, args.gwak_deploy)
    import torch
    from deploy.libs.whiten_utils import BatchWhitener   # gwak's own whitener (wraps ml4gw)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device
    print("[recon] device=%s  data_dir=%s" % (device, args.data_dir), flush=True)

    with h5py.File(args.error_config, "r") as f:
        d = f["data"][:]
    print("[recon] error_config rows=%d" % d.shape[0], flush=True)

    starts, stops, files = build_gps_index(args.data_dir)
    print("[recon] %d raw files, GPS [%d, %d]" % (len(files), starts[0], stops[-1]), flush=True)

    sel, file_idx, T_all, shift_all, info = select_rows(
        d, starts, stops, args.strategy, args.max_events, args.score_thr, seed=1234)
    if args.limit is not None:
        sel = sel[:args.limit]; file_idx = file_idx[:args.limit]
    n = sel.size
    print("[recon] total=%d  no_file=%d  oob=%d  valid=%d  -> selected=%d (strategy=%s)"
          % (info["n_total"], info["n_nofile"], info["n_oob"], info["n_valid_all"], n,
             args.strategy), flush=True)
    if n == 0:
        raise SystemExit("no events selected")

    # NOTE: do NOT cast the module to double. gwak deploy keeps the module in float32 and runs
    # the FIR bandpass in float32; only the PSD and the final whiten are double (cast internally
    # by PsdEstimator/BatchWhitener.forward). Raw strain is cast to float32 first (Sequence does
    # `.astype("float32")`). Matching this is both bit-faithful to deploy and much faster.
    bw = BatchWhitener(kernel_length=KERNEL_LENGTH, sample_rate=SR,
                       inference_sampling_rate=INFER_RATE, batch_size=1,
                       fduration=FDURATION, fftlength=FFTLENGTH,
                       highpass=HIGHPASS).to(device)
    bw.eval()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = np.zeros((n, 2, KERNEL_N), dtype=np.float32)
    meta = {k: np.zeros(n, dtype=np.float64) for k in ("gwak", "shift", "gps", "t0", "ec_row")}
    ok = np.zeros(n, dtype=bool)

    # process grouped by file (open each raw file once)
    order = np.argsort(file_idx)
    t_start = time.time()
    done = 0
    cur_fi = -1; Hfull = Lfull = None
    for pos in order:
        fi = int(file_idx[pos]); ri = int(sel[pos])
        if fi != cur_fi:
            # Load each raw file's H1/L1 ONCE (float32, as deploy's Sequence does) and slice in
            # RAM. Events are grouped by file, so this turns ~TB of per-event sliced disk reads
            # into ~tens of GB of whole-file reads. Peak RAM ~ one file (largest ~1.8 GB f32).
            cur_fi = fi
            with h5py.File(files[fi], "r") as fh:
                Hfull = fh["H1"][:].astype(np.float32)
                Lfull = fh["L1"][:].astype(np.float32)
            nfile = Hfull.shape[0]
            f_a = int(starts[fi])
        T = T_all[ri]; shift = int(shift_all[ri])
        h_s = int(round((T - PRE - f_a) * SR)); h_e = h_s + WINDOW_N
        l_s = int(round((T - PRE + shift - f_a) * SR)); l_e = l_s + WINDOW_N
        if h_s < 0 or l_s < 0 or h_e > nfile or l_e > nfile:
            done += 1
            continue                                        # belt-and-braces (pre-filtered)
        # bandpass runs float32, PSD/whiten are cast to double internally -> matches deploy.
        x = np.stack([Hfull[h_s:h_e], Lfull[l_s:l_e]])[None]   # (1,2,WINDOW_N) float32
        xt = torch.from_numpy(x).to(device)
        with torch.no_grad():
            k = bw(xt).reshape(2, KERNEL_N).detach().cpu().numpy()
        data[pos] = k.astype(np.float32)
        meta["gwak"][pos] = d[ri, C_GWAK]; meta["shift"][pos] = shift
        meta["gps"][pos] = T; meta["t0"][pos] = d[ri, C_T0]; meta["ec_row"][pos] = ri
        ok[pos] = True
        done += 1
        if done % args.progress == 0:
            el = time.time() - t_start
            print("[recon] %d/%d  %.1fs  %.4f s/ev" % (done, n, el, el / done), flush=True)

    nok = int(ok.sum())
    data = data[ok]
    print("[recon] whitened %d/%d events (%.1fs)" % (nok, n, time.time() - t_start), flush=True)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("Timeslide_data", data=data, compression="gzip", compression_opts=4)
        f.create_dataset("Timeslide_snrs", data=np.zeros(nok, dtype=np.float32))
        f.create_dataset("Timeslide_gwak", data=meta["gwak"][ok].astype(np.float32))
        f.create_dataset("Timeslide_shift", data=meta["shift"][ok].astype(np.int32))
        f.create_dataset("Timeslide_gps", data=meta["gps"][ok])
        f.create_dataset("Timeslide_t0", data=meta["t0"][ok])
        f.create_dataset("Timeslide_ec_row", data=meta["ec_row"][ok].astype(np.int64))
        f.attrs.update(dict(
            sample_rate=SR, kernel_length=KERNEL_LENGTH, psd_length=PSD_LENGTH,
            fduration=FDURATION, fftlength=FFTLENGTH, highpass=HIGHPASS, infer_rate=INFER_RATE,
            data_dir=args.data_dir, error_config=args.error_config, strategy=args.strategy,
            detector_order="index0=H1,index1=L1", whitener="gwak BatchWhitener (ml4gw)",
            n_total=info["n_total"], n_nofile=info["n_nofile"], n_oob=info["n_oob"],
            n_valid_all=info["n_valid_all"], n_written=nok))
    print("[recon] wrote %s  Timeslide_data shape=%s" % (args.out, data.shape), flush=True)


if __name__ == "__main__":
    main()
