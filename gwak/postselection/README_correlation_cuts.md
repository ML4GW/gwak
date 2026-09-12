# Correlation-cut postselection (`make_correlation_cuts.py`)

Native GWAK_correlationcuts postselection veto. **Supersedes `cwb_veto_simple.py`**: same
command-line inputs and the same CSV output *skeleton*, but every metric column is a native
tile-space coherence/correlation feature (new names, new formulas) and the headline veto is this
project's fused log-likelihood-ratio metric **`C_full` / `C_coh`** at fixed background-rejection
working points (90% and 99%).

These features are fast PROXIES for cWB/oLIB coherence statistics computed on gwpy Q-transform
tiles — not exact cWB/oLIB statistics. Every cWB/oLIB-inspired quantity carries a `_tile`/`_proxy`
suffix accordingly.

## Layout (all of this must sit together in `gwak/postselection/`)
```
make_correlation_cuts.py            # the cut maker (veto + orchestration)
gwak_corrcuts/                      # feature-math package (numpy + gwpy; NOT pip-installable)
combined_model.json                 # trained C_full / C_coh weights + 90%/99% working points
calibration_hrss.json               # E_thr + kappa_noise (tiles MUST match the trained model)
postselection_thresholds_example.json   # optional extra per-feature {min,max} cuts
recon_timeslide.py                  # ONLY needed for raw --data-dir mode (gwak-venv whitening)
```
The script puts its own directory on `sys.path` and resolves the two JSONs next to itself, so it
runs from any CWD with no `--model-json`/`--calib` flags.

## Input (same as the reference)
A GWAK `error_config.h5` (the `(N,7)` `data` table = `[t0, length, shift, error_start, error_end,
duration, gwak_value]`) plus whitened 1-s kernels for those segments. Kernels come one of two ways:

* `--kernels <h5>` — pre-whitened Stage-1 file from `recon_timeslide.py` (`Timeslide_data`
  `(N,2,4096)` + `Timeslide_ec_row`). **Recommended**; needs no torch.
* `--data-dir <raw dir>` — the reference's raw `background-{t0}-{len}.h5` dir. The script then shells
  `recon_timeslide.py` in the gwak deploy venv (`--gwak-python`) to whiten EXACTLY as gwak did.

## Output (same skeleton as the reference)
```
segment_idx, success, error,
t0, length, shift, error_start, error_end, duration, gwak_value,   # metadata (unchanged)
<native tile features ...>, C_full, C_coh,                         # metrics (native)
veto_C_full_90, veto_C_full_99, veto_C_coh_90, veto_C_coh_99,      # requested working points
vetoed, failed_cuts, n_failed_cuts                                 # veto verdict (unchanged)
```
`C_full`/`C_coh` are continuous log-LR scores; the four `veto_*` flags mark whether the event falls
below the trained threshold that rejects 90%/99% of the background population. `vetoed` is the
aggregate at the chosen headline working point (`--veto-metric` / `--veto-rej`, default C_full@90%),
combined with any optional per-feature cuts from `--thresholds`.

## Run
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python make_correlation_cuts.py \
    --error-config <error_config.h5> \
    --kernels      <recon_timeslide output .h5> \
    --output       correlation_cuts.csv
```
Key options: `--first-n N`, `--workers N`, `--trained-vs {R2_timeslide,R1_bg_glitch}`,
`--veto-metric {C_full,C_coh}`, `--veto-rej {0.90,0.99}`, `--thresholds <json>`.

## Runtime dependencies (`--kernels` path)
`numpy`, `scipy`, `gwpy` (**2.1.3**), `pandas`, `h5py`. No torch / ml4gw on this path.

## IMPORTANT — environment faithfulness
`combined_model.json` and `calibration_hrss.json` were fit against **gwpy 2.1.3** tiles. Running the
feature stage under a different gwpy can shift the tiles and therefore the `C_full`/`C_coh` scores
away from the trained model. Keep the feature stage on gwpy 2.1.3, or re-fit the model after any
change.
