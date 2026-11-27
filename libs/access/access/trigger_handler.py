import glob
import re
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


TRIGGER_SEGMENT_PATTERN = re.compile(r"-(\d+)-(\d+)\.h5$")
CATEGORIES = ("cat1", "cat2", "cat3")


def find_trigger_files(
    trigger_dir: Path | str,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    segment_pattern: re.Pattern | str = TRIGGER_SEGMENT_PATTERN,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Find HDF5 trigger files that overlap one or more requested intervals."""
    trigger_dir = Path(trigger_dir)
    starts = np.asarray(interval_starts, dtype=float).reshape(-1)
    ends = np.asarray(interval_ends, dtype=float).reshape(-1)

    starts = np.asarray([np.min(starts)])
    ends = np.asarray([np.max(ends)])

    if starts.size != ends.size:
        raise ValueError("interval_starts and interval_ends must have equal lengths")
    if starts.size == 0:
        return [], []
    if np.any(ends < starts):
        raise ValueError("Every interval end must be at least its start")

    pattern = (
        segment_pattern
        if hasattr(segment_pattern, "search")
        else re.compile(segment_pattern)
    )
    files = []
    segments = []

    first_directory = int(np.floor(starts.min() / 100000))
    last_directory = int(np.floor(ends.max() / 100000) + 1)
    for directory_number in range(first_directory, last_directory + 1):
        directory = trigger_dir / str(directory_number)
        for filename in sorted(glob.glob(str(directory / "*.h5"))):
            match = pattern.search(filename)
            if match is None:
                continue

            segment_start, duration = map(int, match.groups())
            segment_end = segment_start + duration
            if np.any((starts <= segment_end) & (ends >= segment_start)):
                files.append(filename)
                segments.append((segment_start, segment_end))

    return files, segments


def load_triggers(files: Iterable[Path | str]) -> np.ndarray:
    """Load and concatenate the ``triggers`` datasets from HDF5 files."""
    trigger_arrays = []
    for filename in files:
        with h5py.File(filename, "r") as hdf_file:
            if "triggers" not in hdf_file:
                raise KeyError(f"{filename} does not contain a 'triggers' dataset")
            trigger_arrays.append(hdf_file["triggers"][:])

    if not trigger_arrays:
        return np.empty(0, dtype=[("time", "<f8")])
    return np.concatenate(trigger_arrays)


def select_triggers_in_intervals(
    triggers: np.ndarray,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    time_key: str = "time",
) -> np.ndarray:
    """Select records with time strictly inside any requested interval."""
    if triggers.dtype.names is None or time_key not in triggers.dtype.names:
        raise KeyError(f"Trigger records do not contain a '{time_key}' field")
    starts = np.asarray(interval_starts, dtype=float).reshape(-1)
    ends = np.asarray(interval_ends, dtype=float).reshape(-1)
    if starts.size != ends.size:
        raise ValueError("interval_starts and interval_ends must have equal lengths")
    if starts.size == 0:
        return triggers[:0].copy()

    times = np.asarray(triggers[time_key])
    selected = np.any(
        (starts[:, np.newaxis] < times)
        & (times < ends[:, np.newaxis]),
        axis=0,
    )
    return triggers[selected]


def find_missing_intervals(
    triggers: np.ndarray,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    time_key: str = "time",
) -> list[tuple[float, float]]:
    """Return [start, end) intervals that contain no trigger times."""
    if triggers.dtype.names is not None and time_key in triggers.dtype.names:
        times = np.asarray(triggers[time_key])
    elif len(triggers) == 0:
        times = np.asarray([])
    else:
        raise KeyError(f"Trigger records do not contain a '{time_key}' field")

    starts = np.asarray(interval_starts, dtype=float).reshape(-1)
    ends = np.asarray(interval_ends, dtype=float).reshape(-1)
    if starts.size != ends.size:
        raise ValueError("interval_starts and interval_ends must have equal lengths")
    return [
        (float(start), float(end))
        for start, end in zip(starts, ends)
        if not np.any((start < times) & (times < end))
    ]


def categorize_triggers(
    triggers: np.ndarray,
    segment_dir: Path | str = "/home/shihhong.hsu/gwak/gwak/data/segments",
    categories: Iterable[str] = CATEGORIES,
    time_key: str = "time",
) -> dict[str, np.ndarray]:
    """Split triggers by data-quality category, retaining a ``none`` group."""
    if triggers.dtype.names is None or time_key not in triggers.dtype.names:
        raise KeyError(f"Trigger records do not contain a '{time_key}' field")

    trigger_times = np.asarray(triggers[time_key])
    categorized = {}
    any_category = np.zeros(len(triggers), dtype=bool)
    for category in categories:
        segments = np.asarray(
            np.load(Path(segment_dir) / f"segments.o4.{category}-strain-HL.npy"),
            dtype=float,
        ).reshape(-1, 2)
        if segments.size:
            in_category = np.any(
                (segments[:, 0, np.newaxis] < trigger_times)
                & (trigger_times <= segments[:, 1, np.newaxis]),
                axis=0,
            )
        else:
            in_category = np.zeros(len(triggers), dtype=bool)
        categorized[category] = triggers[in_category]
        any_category |= in_category

    categorized["none"] = triggers[~any_category]
    return categorized


def write_triggers(
    triggers: np.ndarray,
    output_path: Path | str,
    missing_segments: Iterable[tuple[float, float]] = (),
) -> None:
    """Write trigger records and their data quality categories to HDF5."""
    output_path = Path(output_path)
    if output_path.suffix != ".h5":
        raise ValueError("Output path must have a '.h5' suffix")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = np.asarray(
        list(missing_segments),
        dtype=[("start", "<f8"), ("end", "<f8")],
    )

    categorized = categorize_triggers(triggers, categories=CATEGORIES)

    with h5py.File(output_path, "w") as hdf_file:
        hdf_file.create_dataset("triggers", data=triggers)
        hdf_file.create_dataset("missing_segments", data=missing)
        for category in (*CATEGORIES, "none"):
            group = hdf_file.create_group(category)
            group.create_dataset("triggers", data=categorized[category])
