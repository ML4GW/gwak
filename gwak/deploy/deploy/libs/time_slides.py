import re
import math
import logging

from pathlib import Path
from typing import List, Tuple



def segments_from_paths(paths: Path, data_foramt: str="hdf5"):
    fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)")
    files = []
    segments = []
    path_list = paths.glob(f"*.{data_foramt}")
    
    for fname in path_list:
        match = fname_re.search(str(fname))
        if match is None:
            logging.warning(f"Couldn't parse file {fname.path}")

        start = float(match.group("t0"))
        duration = float(match.group("length"))
        stop = start + duration
        files.append(fname)
        segments.append([start, stop])
    return files, segments


def calc_shifts_required(Tb: float, T: float, delta: float) -> int:
    r"""
    Calculate the number of shifts required to generate Tb
    seconds of background.

    Solve:
    $$\sum_{i=1}^{N}(T - i\delta) \geq T_b$$
    for the lowest value of N, where \delta is the
    shift increment.

    TODO: generalize to multiple ifos and negative
    shifts, since e.g. you can in theory get the same
    amount of Tb with fewer shifts if for each shift
    you do its positive and negative. This should just
    amount to adding a factor of 2 * number of ifo
    combinations in front of the sum above.
    """

    discriminant = (delta / 2 - T) ** 2 - 2 * delta * Tb        
    # Add try expect TypeError: to handle discriminant < 0 
    # which means that the provided strain data isn't sufficient 
    # to produce required timeslide 
    N = (T - delta / 2 - discriminant**0.5) / delta
    return math.ceil(N)


def get_num_shifts_from_Tb(
    segments: List[Tuple[float, float]], Tb: float, shift: float
) -> int:
    """
    Calculates the number of required time shifts based on a list
    of background segments and the desired total background duration.
    """
    T = sum([stop - start for start, stop in segments])
    return calc_shifts_required(Tb, T, shift)
