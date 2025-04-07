import json
import argparse
import numpy as np


def intersect(seg1, seg2):
    a, b = seg1
    c, d = seg2
    start, end = max(a, c), min(b, d)

    if start < end:
        return [start, end]

    return None


def find_intersections():

    hanford = np.loadtxt(f'data/configs/segments.o4b-2.h1', dtype=int)
    livingston = np.loadtxt(f'data/configs/segments.o4b-2.l1', dtype=int)

    # there aren't that many segments, so N^2 isn't so bad
    valid_segments = []
    for h_elem in hanford:
        for l_elem in livingston:
            intersection = intersect(h_elem, l_elem)
            if intersection is not None:
                valid_segments.append(intersection)

    return np.array(valid_segments)


def main(save_path):
    '''
    Function which takes the valid segments from both detectors
    and finds an "intersection", i.e. segments where both detectors
    are recording data

    paths are string which point to the corresponding .json files
    '''
    valid_segments = find_intersections()

    np.save(save_path, np.array(valid_segments))


main(snakemake.output[0])