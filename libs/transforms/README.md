# transforms

Preprocessing and feature transforms shared by the GWAK projects.

## H1–L1 correlation features (`transforms/correlation.py`)

Extra latent-space features computed on the same whitened kernel that the embedder sees. They are exposed as two new
`coh_mode`s of `frequency_cos_similarity`, so `precompute_embeddings.py` and `combine_models.py` need no other changes:

```python
from transforms import frequency_cos_similarity, cohmode_to_dim
c = frequency_cos_similarity(batch, mode="corr21")   # (B, 21); "corr15" -> (B, 15); cohmode_to_dim("corr21") == 21
```

Or as a module (all tables are registered buffers; traceable with `torch.jit.trace`):

```python
from transforms import CorrelationFeatures
feat = CorrelationFeatures("corr21").to("cuda").eval()
```

`precompute_embeddings.py --latent latent.npy` also saves the appended latent vector `[embedding | correlations]`
(`(N, 6 + 21)` for `coh_mode=corr21`). The snakemake precompute rules write it as `latent.npy` next to
`embeddings.npy` / `correlations.npy`. The features are not standardised.

**Input**: `(B, 2, 4096)` whitened 1 s kernels at 4096 Hz (index 0 = H1, 1 = L1), as produced by `SignalDataloader`
(FIR bandpass 30–2047 Hz + ml4gw `Whiten`, `fduration` 2 s cropped) or the Triton streaming whitener. Any dtype or
overall scale (each event/detector is standardised internally). Kernels must not contain NaN.

**Output order** (`transforms.CORRELATION_FEATURES[mode]`):

| # | feature | meaning |
|---|---|---|
| 0 | `pearson_0lag` | \|Pearson correlation\| of H1 and L1 at zero lag, full 1 s |
| 1 | `cc_maxlag_1s` | max \|normalised cross-correlation\| over lags ±10.7 ms (±44 samples), full 1 s |
| 2 | `cc_maxlag_win512` | same, maximised over 125 ms windows (hop 62.5 ms) |
| 3 | `cc_band_maxz` | max over 6 octave-like bands (20–2000 Hz) of max-lag \|NCC\| × sqrt(2·bandwidth) |
| 4 | `msc_max` | max Welch magnitude-squared coherence, 20–1024 Hz (Hann, 512/256) |
| 5–7 | `xc_env_w64/128/256` | max-lag \|NCC\| in one window (half-width 15.6 / 31.3 / 62.5 ms) centred on the loudest smoothed H1+L1 Hilbert envelope point (30–1900 Hz) |
| 8 | `xc_env_w256_lag_ms` | \|lag\| (ms) of that maximum for the 62.5 ms window |
| 9 | `ccen_tf` | multi-Q (Q = 4, 11, 32, 90, 256) time-frequency coherence 2\|C\|/(E_H + E_L) over the loud tiles, best lag ±10.7 ms |
| 10 | `ccen_tf_reg` | same with +20 (SNR²) in the denominator (suppresses quiet coincidences) |
| 11 | `wcc2_tf` | \|C\| / sqrt(E_H E_L) |
| 12 | `balance` | \|ln(E_H / E_L)\| of the loud-tile cluster |
| 13 | `dt_peak_ms` | \|time difference\| of the loudest H1 and L1 tiles (ms) |
| 14 | `dlogf_peak` | \|ln(f_H / f_L)\| of those tiles |
| 15 | `ecoh_tf` | \|C\| (coherent energy, SNR² units) — corr21 only from here |
| 16 | `enet_peak` | loudest network tile energy e_H + e_L |
| 17, 18 | `epeak_H`, `epeak_L` | loudest single-detector tile energy |
| 19, 20 | `e_clu_H`, `e_clu_L` | loud-tile cluster energy per detector (SNR² units) |

TF tiles are ml4gw `QTile` rows (tables built once at construction with ml4gw; forward is pure torch). The line rows
(58/59 Hz) and the energy calibration were derived on the O4b one-month whitened timeslides.

**Validation** (torch port vs the validated numpy/ml4gw implementation, 88,985 trusted kernels): max relative difference
1.6e-5 (eager), 3e-4 (traced, after TorchScript fusion). Cost ~1–2 ms per kernel on an RTX PRO 4000 at batch 256.
TorchScript note: `torch.round` inside a traced graph is fused with a different rule for exact halves than eager
PyTorch; the lag shifts are therefore precomputed integer tables. `tests/test_correlation.py` checks traced vs eager over
repeated calls.

**Triton**: models traced with torch 2.12.1 (as `combine_models.py` does) load and run in the ml4gw
`hermes/tritonserver:23.01` image (libtorch 1.14): on an RTX 3080 the served `corr21` output matches eager to 6e-4
(relative, worst kernel) over repeated calls. On Blackwell GPUs (sm_120, e.g. RTX PRO 4000) that image fails at run
time: its libtorch JIT-compiles some complex-number kernels with the CUDA 12.0 NVRTC, which does not know sm_120
(`nvrtc: error: invalid value for --gpu-architecture`). Use pre-Blackwell GPUs or a newer Triton image there.

Tests: `python -m pytest libs/transforms/tests` (or `python libs/transforms/tests/test_correlation.py`).
