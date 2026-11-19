"""Unit tests for transforms.correlation (python -m pytest libs/transforms/tests, or run this file directly)."""
import math

import numpy as np
import torch

from transforms import CORRELATION_FEATURES, CorrelationFeatures, cohmode_to_dim, frequency_cos_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sine_gaussian(n, f0=150.0, q=9.0, lag=20, snr=12.0, seed=0):
    """n kernels of white noise with the same sine-Gaussian in H1 and L1 (L1 delayed by `lag` samples)."""
    t = np.arange(4096) / 4096
    tau = q / (math.sqrt(2) * math.pi * f0)
    s = lambda sh: np.exp(-((t - 0.5 - sh / 4096) / tau) ** 2) * np.sin(2 * math.pi * f0 * (t - sh / 4096))
    sig = np.stack([s(0), s(lag)])
    sig *= snr / np.sqrt((sig[0] ** 2).sum())
    noise = np.random.default_rng(seed).standard_normal((n, 2, 4096))
    return torch.as_tensor(noise + sig[None], device=DEVICE), torch.as_tensor(noise, device=DEVICE)


def test_shapes_and_modes():
    x, _ = _sine_gaussian(5)
    for mode, names in CORRELATION_FEATURES.items():
        out = frequency_cos_similarity(x, mode)
        assert out.shape == (5, len(names)) == (5, cohmode_to_dim(mode))
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
    full = frequency_cos_similarity(x, "corr21")
    assert torch.equal(frequency_cos_similarity(x, "corr15"), full[:, :15])


def test_coherent_signal_scores_higher_than_noise():
    sig, noise = _sine_gaussian(16)
    names = CORRELATION_FEATURES["corr21"]
    fs = frequency_cos_similarity(sig, "corr21").median(0).values
    fn = frequency_cos_similarity(noise, "corr21").median(0).values
    for k in ["ccen_tf", "wcc2_tf", "cc_maxlag_1s", "xc_env_w256", "enet_peak"]:
        i = names.index(k)
        assert fs[i] > fn[i], k
    assert fs[names.index("ccen_tf")] > 0.9


def test_flat_and_scaled_input():
    x, _ = _sine_gaussian(4)
    z = torch.zeros(3, 2, 4096, device=DEVICE, dtype=torch.float64)
    z[1] += 3.0
    z[2, 0] = x[0, 0]
    assert torch.isfinite(frequency_cos_similarity(z, "corr21")).all()
    a = frequency_cos_similarity(x, "corr21")
    # an exact power-of-two rescale (strain-like scale) must give identical features
    assert torch.equal(frequency_cos_similarity(x * 2.0 ** -70, "corr21"), a)
    # a generic rescale changes the input at the 1e-16 level; the TF loud-tile selection may flip for rare kernels
    rel = (frequency_cos_similarity(x * 1e-21, "corr21") - a).abs() / a.abs().clamp(min=1e-3)
    assert (rel > 1e-4).float().mean() < 0.02


def test_traced_matches_eager_over_repeated_calls():
    """torch.jit.trace (as combine_models) must keep matching eager after the TorchScript fuser kicks in."""
    x, _ = _sine_gaussian(64)
    m = CorrelationFeatures("corr21").to(DEVICE).eval()
    with torch.no_grad():
        tr = torch.jit.trace(m, x, check_trace=False)
        for batch in [x, x[:7], x[:1]]:
            e = m(batch)
            for _ in range(4):
                assert ((tr(batch) - e).abs() / e.abs().clamp(min=1e-3)).max() < 1e-4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
