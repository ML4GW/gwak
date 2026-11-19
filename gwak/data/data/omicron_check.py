'''
run example:
cd <gwak_dir>
source .gwak/env.sh
uv run --project /home/shihhong.hsu/gwak/gwak/data/pyproject.toml \
    python /home/shihhong.hsu/gwak/gwak/data/data/omicron_check.py \
        --project-name omicron_check_test \
            --outlier-config ResNet_6d_narrow_band_corr15_NF_from_file=/home/hongyin.chen/Container/GWAK/Outputs/infer/HL/O4b_gwak_cat12/O4b_cat1-loud_glitch/ResNet_6d_narrow_band_selected-9_NF_features-no_context/one_decade/outlier_config.h5 \
            --outlier-config RResNet_6d_narrow_band_corr15_NF_from_file=/home/hongyin.chen/Container/GWAK/Outputs/infer/HL/O4b_gwak_cat12/O4b_cat1-chunked/ResNet_6d_narrow_band_corr15_NF_from_file/one_decade/outlier_config.h5
'''

import argparse
import re
import os
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator

from access.trigger_handler import (
    find_missing_intervals,
    find_trigger_files,
    load_triggers,
    select_triggers_in_intervals,
    write_triggers,
)


HISTOGRAM_RANGES = {
    "frequency": (30, 10000),
    "snr": (4, 3000),
    "time": (1400000000, 1410000000),
}
SEGMENT_PATTERN = re.compile(r"-(\d+)-(\d+)\.h5$")


def load_outlier_config(outlier_path: Path | str) -> dict[str, np.ndarray]:
    """Load a GWAK outlier HDF5 configuration into NumPy arrays."""
    with h5py.File(outlier_path, "r") as hdf_file:
        required = {"event_start", "event_end", "shifts"}
        missing = required.difference(hdf_file.keys())
        if missing:
            raise KeyError(
                f"{outlier_path} is missing datasets: {', '.join(sorted(missing))}"
            )
        return {key: hdf_file[key][:] for key in hdf_file}


def apply_omicron_event_windows(
    outlier_result: Mapping[str, np.ndarray],
    ifo: str,
    check_buffer: float = 5.0,
    correction: float = -1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply GWAK-to-IFO timing offsets and the Omicron comparison buffer."""
    event_starts = np.asarray(outlier_result["event_start"], dtype=float)
    event_ends = np.asarray(outlier_result["event_end"], dtype=float)
    shifts = np.asarray(outlier_result["shifts"], dtype=float)

    if ifo == "H1":
        starts = event_starts - check_buffer
        ends = event_ends + check_buffer
    elif ifo == "L1":
        starts = event_starts + shifts - check_buffer
        ends = event_ends + shifts + check_buffer
    else:
        raise ValueError(f"Unsupported IFO: {ifo}")

    return starts + correction, ends + correction


def omicron_trigger_path(
    trigger_root: Path | str,
    ifo: str,
    channel: str,
) -> Path:
    """Build the standard Omicron archive directory for an IFO/channel."""
    channel_tag = channel.replace(":", "_")
    return Path(trigger_root) / ifo / f"{channel_tag}_OMICRON"


def plot_trigger_comparison(
    triggers: np.ndarray,
    outlier_triggers: np.ndarray,
    ifo: str,
    channel: str,
    segment_time: float,
    outlier_time: float,
    output_path: Path | str,
) -> None:
    """Save normalized Omicron histograms and an SNR-frequency scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    histogram_names = tuple(HISTOGRAM_RANGES)
    figure = plt.figure(figsize=(14, 7))
    grid = figure.add_gridspec(2, 3, height_ratios=(1, 1))
    axes = [figure.add_subplot(grid[0, 0])]
    axes.extend(
        figure.add_subplot(grid[0, column], sharey=axes[0])
        for column in range(1, 3)
    )
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    for name, axis in zip(histogram_names, axes):
        bins = np.geomspace(*HISTOGRAM_RANGES[name], 20)
        trigger_counts, _ = np.histogram(triggers[name], bins=bins)
        outlier_counts, _ = np.histogram(outlier_triggers[name], bins=bins)

        axis.stairs(
            trigger_counts / len(triggers) if len(triggers) else np.zeros_like(trigger_counts),
            bins,
            label="omicron",
        )
        if len(outlier_triggers):
            axis.stairs(
                outlier_counts / len(outlier_triggers),
                bins,
                label="outlier",
            )
        else:
            axis.plot([], [], label="outlier (no triggers)")
        axis.set_xlabel(name)
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10))
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10))
        axis.grid(True, axis="x", which="major")
        axis.grid(True, axis="y", which="major")

    axes[0].set_ylabel("ratio")
    axes[1].set_title(f"Normalized Histogram: {ifo}:{channel}")
    axes[-1].legend()

    scatter_axis = figure.add_subplot(grid[1, :])
    scatter_axis.scatter(
        triggers["frequency"],
        triggers["snr"],
        s=10,
        alpha=0.5,
        label=f"omicron: {len(triggers)} trg. in {segment_time:.0f} s",
    )
    scatter_axis.scatter(
        outlier_triggers["frequency"],
        outlier_triggers["snr"],
        s=10,
        label=f"outlier: {len(outlier_triggers)} trg. in {outlier_time:.0f} s",
    )
    scatter_axis.set_title("SNR - Frequency")
    scatter_axis.set_xlabel("frequency [Hz]")
    scatter_axis.set_ylabel("snr")
    scatter_axis.xaxis.set_major_locator(MultipleLocator(1000))
    scatter_axis.set_ylim(5, 300)
    scatter_axis.set_yscale("log")
    scatter_axis.yaxis.set_major_locator(LogLocator(base=10))
    scatter_axis.grid(True, which="major")
    scatter_axis.legend()

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path)
    plt.close(figure)


def run_omicron_check(
    outlier_configs: Mapping[str, Path | str],
    out_dir: Path | str,
    project_name: str,
    trigger_root: Path | str = "/home/detchar/triggers",
    ifos: Iterable[str] = ("H1", "L1"),
    channel: str = "GDS_CALIB_STRAIN_NOLINES",
    check_buffer: float = 5.0,
    correction: float = -1.25,
) -> None:
    """Compare GWAK outlier windows against archived Omicron triggers."""
    out_dir = Path(out_dir)
    for outlier_name, outlier_path in outlier_configs.items():
        outlier_result = load_outlier_config(outlier_path)
        output_dir = out_dir / "omicron_check" / project_name / outlier_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for ifo in ifos:
            interval_starts, interval_ends = apply_omicron_event_windows(
                outlier_result,
                ifo,
                check_buffer=check_buffer,
                correction=correction,
            )
            trigger_dir = omicron_trigger_path(trigger_root, ifo, channel)
            files, segments = find_trigger_files(
                trigger_dir,
                interval_starts,
                interval_ends,
                segment_pattern=SEGMENT_PATTERN,
            )
            if not files:
                raise FileNotFoundError(
                    f"No Omicron trigger files for {ifo}:{channel} overlap "
                    f"{outlier_name}"
                )

            triggers = load_triggers(files)
            outlier_triggers = select_triggers_in_intervals(
                triggers,
                interval_starts,
                interval_ends,
            )
            segment_time = sum(end - start for start, end in segments)
            outlier_time = sum(ends - starts for starts, ends in zip(interval_starts, interval_ends))

            plot_trigger_comparison(
                triggers,
                outlier_triggers,
                ifo,
                channel,
                segment_time,
                outlier_time,
                output_dir / f"{ifo}:{channel}.png",
            )
            missing = find_missing_intervals(
                outlier_triggers,
                interval_starts,
                interval_ends,
            )
            write_triggers(
                outlier_triggers,
                output_dir / f"{ifo}_triggers.h5",
                missing,
            )
            print(
                f"{outlier_name} {ifo}: "
                f"{len(outlier_triggers)} outlier triggers, "
                f"{len(missing)} missing intervals"
            )


def parse_config_pairs(values: Iterable[str]) -> dict[str, Path]:
    configs = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"Outlier config must use NAME=PATH: {value}"
            )
        name, path = value.split("=", 1)
        configs[name] = Path(path)
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GWAK outliers against Omicron triggers."
    )
    parser.add_argument(
        "--outlier-config",
        action="append",
        required=True,
        help="NAME=/path/to/outlier_config.h5 (repeatable)",
    )
    parser.add_argument("--project-name", default="omicron_check")
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ["GWAK_OUTPUT_DIR"]))
    parser.add_argument("--trigger-root", type=Path, default=Path("/home/detchar/triggers"))
    parser.add_argument("--ifos", nargs="+", default=["H1", "L1"])
    parser.add_argument("--channel", default="GDS_CALIB_STRAIN_NOLINES")
    parser.add_argument("--check-buffer", type=float, default=5.0)
    parser.add_argument("--correction", type=float, default=-1.25)
    args = parser.parse_args()
    args.outlier_configs = parse_config_pairs(args.outlier_config)
    return args


def main():
    args = parse_args()
    run_omicron_check(
        outlier_configs=args.outlier_configs,
        out_dir=args.out_dir,
        project_name=args.project_name,
        trigger_root=args.trigger_root,
        ifos=args.ifos,
        channel=args.channel,
        check_buffer=args.check_buffer,
        correction=args.correction,
    )


if __name__ == "__main__":
    main()
