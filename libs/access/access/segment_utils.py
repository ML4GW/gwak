import logging
import numpy as np
from pathlib import Path
from typing import Optional


def load_segments(segments):
    """Load an (N, 2) [start, end] array from a .npy or text file."""
    segments = Path(segments)
    if segments.suffix == ".npy":
        return np.load(segments).reshape(-1, 2)
    return np.loadtxt(segments, ndmin=2).reshape(-1, 2)


def intersect_two_lists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the overlaps of two *sorted* (N, 2) [start, end] arrays.
    O(n + m): advance whichever list's segment ends first.
    """
    i = j = 0
    out = []

    while i < len(a) and j < len(b):
        start = max(a[i, 0], b[j, 0])
        end   = min(a[i, 1], b[j, 1])

        if start < end:
            out.append([start, end])

        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1

    return np.asarray(out).reshape(-1, 2)


def find_local_segments(
    segment_dir: Path,
    segment_type: str,
    ifos: list[str]
) -> Optional[np.ndarray]:
    """
    Intersect the per-IFO segment lists
        <segment_dir>/segments.<segment_type>.<IFO>
    (e.g. segments.original.o4b-2.H1) into the segments where all
    `ifos` are recording. Returns None if any IFO's file is missing.
    """
    paths = [Path(segment_dir) / f"segments.{segment_type}.{ifo}" for ifo in ifos]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        logging.info(f"No local segment list {missing}")
        return None

    valid = None
    for path in paths:
        logging.info(f"Loading segments from {path}")
        segs = load_segments(path)
        segs = segs[np.argsort(segs[:, 0])]
        valid = segs if valid is None else intersect_two_lists(valid, segs)

    return valid


def write_segment_list(segs, resolved_segments: Path):
    """
    Save the segments as an (N, 2) int array of [start, end], either as
    .npy or as text with one "start end" line per segment. Row N
    (0-based) is the segment processed by `get_strain --seg_index N`.
    """
    resolved_segments = Path(resolved_segments)
    resolved_segments.parent.mkdir(parents=True, exist_ok=True)
    rows = [(int(np.ceil(s)), int(np.floor(e))) for s, e in segs]
    rows = np.asarray(rows, dtype=int).reshape(-1, 2)
    if resolved_segments.suffix == ".npy":
        np.save(resolved_segments, rows)
    else:
        np.savetxt(resolved_segments, rows, fmt="%d")
    logging.info(f"Wrote {len(rows)} segment(s) to {resolved_segments}")


def select_segments(segs, seg_index: Optional[int] = None):
    if seg_index is None:
        return segs
    if not 0 <= seg_index < len(segs):
        raise IndexError(
            f"--seg_index {seg_index} out of range, "
            f"there are {len(segs)} segment(s)"
        )
    return [segs[seg_index]]

# Old version of segment list production
def find_intersections(ifos, segment_type, folder_segments) -> np.ndarray:
    """
    Intersect the segment lists for ALL IFOs given in
    `snakemake.wildcards.ifos` (e.g. 'hlv').
    """
    ifos = list(ifos)        # e.g. ['h','l','v']

    # load every detector’s list
    segment_lists = [load_segments(ifo, segment_type, folder_segments) for ifo in ifos]

    # iterative k-way intersection
    valid = segment_lists[0]
    for segs in segment_lists[1:]:
        valid = intersect_two_lists(valid, segs)
        if len(valid) == 0:                      # nothing left
            break

    return valid