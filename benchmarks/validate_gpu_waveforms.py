"""
Compare GPU waveform generation with the existing CPU path.

For each supported class, compare the detector responses against
a float64 reference. WhiteNoiseBurst is checked using total power
because it generates random noise internally.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gwak/train/train"))


def load_datamodule(configs, data_dir, extra):
    from cli import GwakMultiSignalCLI

    class NoFit(GwakMultiSignalCLI):
        def _run_subcommand(self, subcommand):
            pass

    args = ["fit"]
    for c in configs:
        args += ["--config", c]
    if data_dir:
        args += ["--data.init_args.data_dir", data_dir]
    # always start from the original code path
    args += ["--trainer.logger", "false",
             "--data.init_args.gpu_waveforms", "null"] + extra
    sys.argv = sys.argv[:1]
    return NoFit(args=args, save_config_callback=None).datamodule


def rel_err(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm())


def float64_reference(dl, signal_class, waveform, config, ifos, params, ra, dec, phic):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # some ml4gw constants use the default dtype
    try:
        waveform = copy.deepcopy(waveform).cpu().double()
        p = {}
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                v = torch.as_tensor(v)
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.double()
            p[k] = v
        tensors, vertices = dl.get_ifo_geometry(*ifos)
        cross, plus = waveform(**p)
        sr = config["sample_rate"]
        if signal_class == "BBH":
            n = int(config["ringdown_duration"] * sr)
            cross = torch.roll(torch.fft.irfft(cross) * sr, -n, dims=-1)
            plus = torch.roll(torch.fft.irfft(plus) * sr, -n, dims=-1)
            psi = p["phic"]
        else:
            psi = phic.double()
        return dl.compute_observed_strain(
            dec.double(), psi, ra.double(), tensors.double(), vertices.double(), sr,
            cross=cross, plus=plus,
        )
    finally:
        torch.set_default_dtype(old_dtype)


def check_waveforms(dm, dl, device, seed, repeats):
    results = {}
    for i, signal_class in enumerate(dm.signal_classes):
        if signal_class in dl.NON_WAVEFORM_CLASSES:
            continue
        generate = dl.generate_waveforms_bbh if signal_class == "BBH" else dl.generate_waveforms_standard
        n = dm.num_per_class[i]
        prior, waveform, config = dm.priors[i], dm.waveforms[i], dm.signal_configs[i]
        rows = []
        for r in range(repeats):
            torch.manual_seed(seed + 100 * i + r)
            params = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                      for k, v in prior.sample(n).items()}
            ra = dm.ra_prior.sample((n,))
            dec = dm.dec_prior.sample((n,))
            args = (n, prior, waveform, dm, config, dm.ifos)

            # same seed before each call, so phic (sampled inside) is the same
            torch.manual_seed(seed + r)
            cpu = generate(*args, parameters=params, ra=ra, dec=dec)
            torch.manual_seed(seed + r)
            gpu = generate(n, prior, copy.deepcopy(waveform), dm, config, dm.ifos,
                           parameters=params, ra=ra, dec=dec, device=device)

            row = {"finite": bool(torch.isfinite(gpu[0]).all())}
            if signal_class == "WhiteNoiseBurst":
                row["power_ratio"] = float((gpu[0].double().cpu() ** 2).sum() / (cpu[0].double().cpu() ** 2).sum())
            else:
                ref = float64_reference(dl, signal_class, waveform, config, dm.ifos, params, ra, dec, cpu[4])
                row["cpu_err"] = rel_err(cpu[0], ref)
                row["gpu_err"] = rel_err(gpu[0], ref)
                row["gpu_vs_cpu"] = rel_err(gpu[0], cpu[0])
            rows.append(row)

        if signal_class == "WhiteNoiseBurst":
            ok = all(r["finite"] and 0.5 < r["power_ratio"] < 2 for r in rows)
        else:
            ok = all(
                row["finite"] and row["gpu_err"] <= 2 * row["cpu_err"] + 1e-6
                for row in rows
            )
        results[signal_class] = {"ok": ok, "rows": rows}
    return results


def main():
    parser = argparse.ArgumentParser(description="compare gpu_waveforms to the original code")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda", help="cpu only checks that the code runs")
    parser.add_argument("extra", nargs="*", help="extra cli arguments, after --")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available")
    device = torch.device(args.device)

    dm = load_datamodule(args.config, args.data_dir, args.extra)
    import dataloader as dl

    waveforms = check_waveforms(dm, dl, device, args.seed, args.repeats)
    passed = all(r["ok"] for r in waveforms.values())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"device": str(device), "passed": passed, "waveforms": waveforms}, indent=2))

    print(f"\nwaveforms on {device} (relative error of the detector response vs float64)")
    for c, r in waveforms.items():
        if c == "WhiteNoiseBurst":
            ratios = ", ".join(f"{row['power_ratio']:.3f}" for row in r["rows"])
            print(f"  {c:16s} power ratio gpu/cpu: {ratios:28s} {'ok' if r['ok'] else 'FAILED'}")
        else:
            cpu_err = max(row["cpu_err"] for row in r["rows"])
            gpu_err = max(row["gpu_err"] for row in r["rows"])
            print(f"  {c:16s} cpu {cpu_err:.2e}  gpu {gpu_err:.2e}  {'ok' if r['ok'] else 'FAILED'}")
    print("all checks passed" if passed else "SOME CHECKS FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
