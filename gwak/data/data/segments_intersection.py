"""
Find the segments in which *all* requested IFOs are simultaneously
recording data.

The script is invoked from Snakemake, which supplies a wildcard
`{ifos}` such as  "hl", "hv", "lv", or "hlv".  Each character
corresponds to an IFO (h → H1, l → L1, v → V1).

The data files follow the pattern
    data/configs/segments.original.o4b-2.<IFO>1
and contain N×2 integer arrays:
    [GPS_START  GPS_END]

The algorithm works as follows:

1.  Load every detector’s segment list.
2.  **Iteratively** intersect the lists until only segments that are
    common to *all* detectors remain.  The two-list intersection is
    O(n+m) and we perform it (|IFOs|−1) times, so it stays fast even
    for many detectors.
"""

import numpy as np
import argparse

def intersect_two_lists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the intersections of two *sorted* [start, end] arrays.

    Both `a` and `b` must be shaped (N,2).  The result is also (M,2),
    where each row is a non-empty overlap of one element from `a`
    and one element from `b`.
    """
    i = j = 0
    out = []

    while i < len(a) and j < len(b):
        start = max(a[i, 0], b[j, 0])
        end   = min(a[i, 1], b[j, 1])

        if start < end:                         # non-zero overlap
            out.append([start, end])

        # advance the list whose segment ends first
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1

    return np.asarray(out, dtype=int)


def load_segments(ifo_char, segment_type, folder_segments) -> np.ndarray:
    """
    Load the segment list for a single IFO, sort it by start time, and
    return a (N,2) int array.
    """
    path = f"{folder_segments}segments.{segment_type}.{ifo_char}1"
    print(f"Loading segments from {path}")
    segs = np.loadtxt(path, dtype=int)
    return segs[np.argsort(segs[:, 0])]          # ensure sorted


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find intersections between given files.'
    )
    parser.add_argument('--folder-segments', type=str, help='Path to the folder containing JIT embedder model.')
    parser.add_argument('--segment-type', type=str, help='Path to the folder containing JIT metric model.')
    parser.add_argument('--ifos', type=str, help='Path to the folder containing JIT metric model.')
    parser.add_argument('--save-path', type=str, help='Path to the folder containing JIT metric model.')
    args = parser.parse_args()
    """
    Write the resulting intersections to `save_path` as an .npy file so
    that downstream Snakemake rules can consume them.
    """
    intersections = find_intersections(args.ifos, args.segment_type, args.folder_segments)
    np.save(args.save_path, intersections)
    print(f"Saved {len(intersections)} intersecting segment(s) to {args.save_path}")
