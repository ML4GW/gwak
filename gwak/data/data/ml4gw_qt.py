#!/usr/bin/env python

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from ml4gw.transforms.qtransform import QScan


SEGMENT_PATTERN = re.compile(r"-(\d+)-(\d+)\.gwf$")


def read_triggers(path: Path) -> dict[str, np.ndarray]:
    """Read trigger records from an Omicron-check HDF5 trigger file."""
    if not path.is_file():
        raise FileNotFoundError(f"Trigger file does not exist: {path}")
    with h5py.File(path, "r") as trigger_file:
        if "triggers" not in trigger_file:
            raise KeyError(f"{path} does not contain a 'triggers' dataset")
        triggers = trigger_file["triggers"][:]
        if triggers.dtype.names is None or "time" not in triggers.dtype.names:
            raise KeyError(f"{path} trigger records do not contain a 'time' field")
        return {
            name: np.asarray(triggers[name]).copy()
            for name in triggers.dtype.names
        }


def get_source(
    start_time: float,
    end_time: float,
    ifo: str,
    hoft_dir: Path,
) -> list[str]:
    """Find frame files overlapping the requested GPS interval."""
    frame_dir = hoft_dir / ifo
    source = []
    for filename in sorted(glob.glob(str(frame_dir / "*.gwf"))):
        match = SEGMENT_PATTERN.search(filename)
        if match is None:
            continue
        segment_start, duration = map(int, match.groups())
        segment_end = segment_start + duration
        if segment_end > start_time and segment_start < end_time:
            source.append(filename)

    if not source:
        candidates = sorted(frame_dir.rglob("*.gwf"))
        for filename in candidates:
            match = SEGMENT_PATTERN.search(str(filename))
            if match is None:
                continue
            segment_start, duration = map(int, match.groups())
            segment_end = segment_start + duration
            if segment_end > start_time and segment_start < end_time:
                source.append(str(filename))

    if not source:
        raise FileNotFoundError(
            f"No {ifo} frame files overlap [{start_time}, {end_time}) "
            f"under {frame_dir}"
        )
    return source


def timeseries_preprocess(
    trigger_time: float,
    ifo: str,
    channel: str,
    hoft_dir: Path,
    sample_rate: int,
    fft_length: float,
    whiten_length: float,
    duration: float,
    buffer_seconds: float,
) -> np.ndarray:
    """Whiten the surrounding data and return exactly duration seconds."""
    samples_per_trigger = round(duration * sample_rate)
    if not np.isclose(samples_per_trigger, duration * sample_rate):
        raise ValueError("duration * sample_rate must be an integer")

    analysis_start = trigger_time - duration / 2
    analysis_end = trigger_time + duration / 2
    read_start = analysis_start - whiten_length - 2 * buffer_seconds
    read_end = analysis_end + 2 * buffer_seconds
    source = get_source(read_start, read_end, ifo, hoft_dir)

    time_series = TimeSeries.read(
        source=source,
        channel=f"{ifo}:{channel}",
        start=read_start,
        end=read_end,
    ).resample(sample_rate)

    asd_end = read_start + whiten_length
    asd_series = time_series.crop(read_start, asd_end)
    asd = asd_series.asd(fftlength=fft_length, overlap=fft_length / 2)
    asd_values = np.asarray(asd.value)
    valid_asd = np.isfinite(asd_values) & (asd_values > 0)
    if not np.any(valid_asd):
        raise ValueError(
            f"ASD contains no positive finite values for {ifo} at "
            f"GPS {trigger_time}"
        )

    whitened = time_series.whiten(asd=asd[valid_asd])
    whitened = whitened.crop(
        read_start + whiten_length + buffer_seconds,
        read_end - buffer_seconds,
    )

    samples = np.asarray(whitened.value)
    sample_times = np.asarray(whitened.times.value)
    if samples.size < samples_per_trigger:
        raise ValueError(
            f"Whitened series is too short for {ifo} at GPS {trigger_time}: "
            f"expected at least {samples_per_trigger} samples, got {samples.size}"
        )

    center_index = int(np.argmin(np.abs(sample_times - trigger_time)))
    first_index = center_index - samples_per_trigger // 2
    last_index = first_index + samples_per_trigger
    if first_index < 0 or last_index > samples.size:
        raise ValueError(
            f"Cannot extract a centered {duration}-second window for "
            f"{ifo} at GPS {trigger_time}"
        )

    window = np.asarray(
        samples[first_index:last_index],
        dtype=np.float32,
    )
    if window.size != samples_per_trigger:
        raise RuntimeError(
            f"Expected {samples_per_trigger} samples, got {window.size}"
        )
    if not np.all(np.isfinite(window)):
        raise ValueError(
            f"Whitened window contains non-finite values for {ifo} at "
            f"GPS {trigger_time}"
        )
    return window


def write_outlier_qtransforms(
    outlier_configs: Mapping[str, Path | str],
    project_name: str,
    out_dir: Path | str | None = None,
    ifos: Iterable[str] = ("H1", "L1"),
    channel: str = "GDS-CALIB_STRAIN_CLEAN",
    hoft_dir: Path | str = "/ceph/mirror/frames/O4/hoft_C00",
    sample_rate: int = 4096,
    fft_length: float = 2,
    whiten_length: float = 60,
    duration: float = 1,
    buffer_seconds: float = 1,
    spectrogram_shape: tuple[int, int] = (256, 256),
) -> None:
    """Create Q-transform spectrograms for triggers produced by Omicron check."""
    if out_dir is None:
        out_dir = Path(os.environ["GWAK_OUTPUT_DIR"])
    out_dir = Path(out_dir)
    hoft_dir = Path(hoft_dir)
    ifos = tuple(ifos)
    samples_per_trigger = round(duration * sample_rate)
    if not np.isclose(samples_per_trigger, duration * sample_rate):
        raise ValueError("duration * sample_rate must be an integer")

    qscan = QScan(
        sample_rate=sample_rate,
        duration=duration,
        spectrogram_shape=spectrogram_shape,
    )

    for outlier_name in outlier_configs:
        input_dir = out_dir / "omicron_check" / project_name / outlier_name
        output_path = input_dir / "qtransform.h5"
        input_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as output_file:
            output_file.attrs["sample_rate_hz"] = sample_rate
            output_file.attrs["duration_seconds"] = duration
            output_file.attrs["whitening_length_seconds"] = whiten_length
            output_file.attrs["buffer_seconds"] = buffer_seconds
            output_file.attrs["channel"] = channel
            output_file.attrs["spectrogram_shape"] = spectrogram_shape

            for ifo in ifos:
                trigger_path = input_dir / f"{ifo}_triggers.h5"
                triggers = read_triggers(trigger_path)
                trigger_times = triggers["time"]
                n_triggers = len(trigger_times)
                print(f"{outlier_name}: processing {n_triggers} {ifo} triggers")

                ifo_group = output_file.create_group(ifo)
                trigger_group = ifo_group.create_group("trigger_info")
                for name, values in triggers.items():
                    trigger_group.create_dataset(name, data=values)

                spectrogram_options = {}
                if n_triggers:
                    spectrogram_options = {
                        "chunks": (1, *spectrogram_shape),
                        "compression": "gzip",
                        "shuffle": True,
                    }
                spectrograms = ifo_group.create_dataset(
                    "q_spectrograms",
                    shape=(n_triggers, *spectrogram_shape),
                    dtype=np.float32,
                    **spectrogram_options,
                )
                spectrograms.attrs["axes"] = "trigger, frequency, time"
                chosen_q = ifo_group.create_dataset(
                    "chosen_q", shape=(n_triggers,), dtype=np.float32
                )
                frequency_ranges = ifo_group.create_dataset(
                    "frequency_range_hz",
                    shape=(n_triggers, 2),
                    dtype=np.float32,
                )
                frequency_bins = ifo_group.create_dataset(
                    "frequency_bins_hz",
                    shape=(n_triggers, spectrogram_shape[0]),
                    dtype=np.float32,
                )

                for index, trigger_time in enumerate(trigger_times):
                    waveform = timeseries_preprocess(
                        float(trigger_time),
                        ifo,
                        channel,
                        hoft_dir,
                        sample_rate,
                        fft_length,
                        whiten_length,
                        duration,
                        buffer_seconds,
                    )
                    input_tensor = torch.from_numpy(waveform)[None, None, :]
                    result = qscan(input_tensor)

                    scores = torch.stack(
                        [transform.get_max_energy() for transform in qscan.q_transforms]
                    )
                    best_transform = qscan.q_transforms[int(scores.argmax().item())]
                    spectrograms[index] = result.reshape(spectrogram_shape).detach().cpu().numpy()

                    fmin, fmax = map(float, best_transform.frange)
                    chosen_q[index] = best_transform.q
                    frequency_ranges[index] = (fmin, fmax)
                    frequency_bins[index] = torch.logspace(
                        np.log10(fmin),
                        np.log10(fmax),
                        steps=spectrogram_shape[0],
                    ).cpu().numpy()

                    if (index + 1) % 25 == 0 or index + 1 == n_triggers:
                        print(f"  {ifo}: {index + 1}/{n_triggers}")

            time_bins = output_file.create_dataset(
                "time_bins_seconds",
                data=np.arange(spectrogram_shape[1], dtype=np.float32)
                * duration
                / spectrogram_shape[1],
            )
            time_bins.attrs["description"] = "Seconds from the start of each window"

        print(f"Saved Q-transform data to {output_path}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ml4gw Q-transform spectrograms for Omicron-check triggers."
    )
    parser.add_argument(
        "--outlier-config",
        action="append",
        required=True,
        help="NAME=/path/to/outlier_config.h5 (repeatable)",
    )
    parser.add_argument("--project-name", default="omicron_check_test")
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ["GWAK_OUTPUT_DIR"]))
    parser.add_argument("--ifos", nargs="+", default=["H1", "L1"])
    parser.add_argument("--channel", default="GDS_CALIB_STRAIN_CLEAN")
    parser.add_argument("--hoft-dir", type=Path, default=Path("/ceph/mirror/frames/O4/hoft_C00"))
    parser.add_argument("--sample-rate", type=int, default=4096)
    parser.add_argument("--fft-length", type=float, default=2)
    parser.add_argument("--whiten-length", type=float, default=60)
    parser.add_argument("--duration", type=float, default=1)
    parser.add_argument("--buffer-seconds", type=float, default=1)
    args = parser.parse_args()
    args.outlier_configs = parse_config_pairs(args.outlier_config)
    return args


def main() -> None:
    args = parse_args()
    write_outlier_qtransforms(
        outlier_configs=args.outlier_configs,
        project_name=args.project_name,
        out_dir=args.out_dir,
        ifos=args.ifos,
        channel=args.channel,
        hoft_dir=args.hoft_dir,
        sample_rate=args.sample_rate,
        fft_length=args.fft_length,
        whiten_length=args.whiten_length,
        duration=args.duration,
        buffer_seconds=args.buffer_seconds,
    )


if __name__ == "__main__":
    main()
