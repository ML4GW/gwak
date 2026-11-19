import glob
import re
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


TRIGGER_SEGMENT_PATTERN = re.compile(r"-(\d+)-(\d+)\.h5$")


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


def write_triggers(
    triggers: np.ndarray,
    output_path: Path | str,
    missing_segments: Iterable[tuple[float, float]] = (),
) -> None:
    """Write trigger records to a HDF5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = np.asarray(list(missing_segments), dtype=[("start", "<f8"), ("end", "<f8")])

    segment_dir = '/home/shihhong.hsu/gwak/gwak/data/segments'
    cats = ['cat1', 'cat2', 'cat3']
    cats_path = { cat :f'{segment_dir}/segments.o4.{cat}-strain-HL.npy' for cat in cats }
    cat_segments = { cat: np.load(cats_path[cat]) for cat in cats }
    triggers_cat = {}
    
    trigger_dtype = triggers.dtype
    for cat in cats + ['none']:
        triggers_cat[cat] = np.asarray([], dtype=trigger_dtype)

    for trigger in triggers:
        is_none = False
        t_trigger = trigger[0]
        for cat in cats:
            segments = cat_segments[cat]
            cat_st = np.asarray([ t for t,_ in segments ])
            cat_ed = np.asarray([ t for _,t in segments ])
            cond_st = cat_st < t_trigger
            cond_ed = t_trigger <= cat_ed
            cond_cat = any([ cond_st[i] and cond_ed[i] for i in range(len(cond_st)) ])
            is_none = is_none or cond_cat
            if cond_cat:
                triggers_cat[cat] = np.append(triggers_cat[cat], trigger)
        if not is_none:
            triggers_cat['none'] = np.append(triggers_cat['none'], trigger)


    if output_path.suffix == ".h5":
        with h5py.File(output_path, "w") as hdf_file:
            hdf_file.create_dataset("triggers", data=triggers)
            hdf_file.create_dataset("missing_segments", data=missing)

            for cat in cats + ['none']:
                group_cat = hdf_file.create_group(cat)
                group_cat.create_dataset('triggers', data=triggers_cat[cat])


    else:
        raise ValueError("Output path must have a '.txt' or '.h5' suffix")
