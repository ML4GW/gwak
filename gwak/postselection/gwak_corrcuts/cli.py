"""cli.py — argparse entry point. Sets the thread cap BEFORE importing numpy.

Usage:
    python3 -m gwak_corrcuts.cli extract --infile <h5> --out <csv> [--subsample N] [--classes ...]
"""
import os

# MANDATORY (archaeology §D): cap BLAS/OMP/MKL threads before numpy is imported,
# otherwise numpy reductions segfault under RLIMIT_NPROC on /usr/bin/python3.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402


def _build_config(args):
    from gwak_corrcuts.config import Config
    return Config(
        regime=args.regime,
        subsample=args.subsample,
        batch_size=args.batch_size,
        sig_energy_thr=args.sig_energy_thr,
        seed=args.seed,
        output_format=args.output_format,
    )


def main(argv=None):
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "thread cap not set"
    parser = argparse.ArgumentParser(prog="gwak_corrcuts")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("extract", help="extract slice features over a (subsampled) dataset")
    p.add_argument("--infile", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--regime", default="A", choices=["A", "B"])
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--sig-energy-thr", dest="sig_energy_thr", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-format", dest="output_format", default="csv",
                   choices=["csv", "parquet"])
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--calibrate", action="store_true",
                   help="calibrate E_thr + kappa_noise on Background before extracting")

    c = sub.add_parser("calibrate", help="calibrate E_thr + kappa_noise on the Background class")
    c.add_argument("--infile", required=True)
    c.add_argument("--regime", default="A", choices=["A", "B"])
    c.add_argument("--n-ethr", dest="n_ethr", type=int, default=200)
    c.add_argument("--ethr-quantile", dest="ethr_quantile", type=float, default=0.999)
    c.add_argument("--n-kappa", dest="n_kappa", type=int, default=400)
    c.add_argument("--seed", type=int, default=1234)

    pl = sub.add_parser("plot", help="make the three views per feature from a feature table")
    pl.add_argument("--table", required=True)
    pl.add_argument("--outdir", default="plots")

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.error("a subcommand is required (e.g. 'extract')")

    if args.cmd == "extract":
        from gwak_corrcuts.pipeline import run
        from gwak_corrcuts.calibrate import calibrate
        cfg = _build_config(args)
        if getattr(args, "calibrate", False):
            cal = calibrate(args.infile, cfg)
            print("calibrated: E_thr=%.4f kappa_noise=%s E[chi2]=%s"
                  % (cal["e_thr"], cal["kappa_noise"], cal["e_chi2_check"]))
        rows = run(args.infile, cfg, args.out, classes=args.classes)
        print("wrote %d rows to %s (regime=%s)" % (len(rows), args.out, cfg.regime_label))
        return 0
    if args.cmd == "calibrate":
        from gwak_corrcuts.config import Config
        from gwak_corrcuts.calibrate import calibrate
        cfg = Config(regime=args.regime, seed=args.seed)
        cal = calibrate(args.infile, cfg, n_events_ethr=args.n_ethr,
                        ethr_quantile=args.ethr_quantile, n_events_kappa=args.n_kappa,
                        seed=args.seed)
        import json
        print(json.dumps(cal, indent=2))
        return 0
    if args.cmd == "plot":
        from gwak_corrcuts.io import read_feature_table
        from gwak_corrcuts.plotting import make_all_plots, gated_fraction_table
        df, cfg = read_feature_table(args.table)
        made, feats = make_all_plots(df, cfg, args.outdir)
        print("made %d plot files over %d features -> %s/" % (len(made), len(feats), args.outdir))
        for r in gated_fraction_table(df):
            print("  %-16s n=%-6d cc_NaN=%.3f low_occ=%.3f" % r)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
