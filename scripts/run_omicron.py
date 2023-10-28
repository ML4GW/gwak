"""
Script that generates a dataset of glitches from omicron triggers.
Taken from https://github.com/ML4GW/BBHNet/blob/main/projects/sandbox/datagen/datagen/scripts/glitches.py
"""
import time
import argparse
import configparser
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
from omicron.cli.process import main as omicron_main

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SNR_THRESH,
    STRAIN_START,
    STRAIN_STOP,
    Q_MIN,
    Q_MAX,
    F_MIN,
    CLUSTER_DT,
    CHUNK_DURATION,
    SEGMENT_DURATION,
    OVERLAP,
    MISTMATCH_MAX,
    WINDOW,
    CHANNEL,
    FRAME_TYPE,
    GLITCH_SAMPLE_RATE,
    STATE_FLAG,
    IFOS)


def omicron_main_wrapper(
    start: int,
    stop: int,
    run_dir: Path,
    q_min: float,
    q_max: float,
    f_min: float,
    f_max: float,
    sample_rate: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    snr_thresh: float,
    frame_type: str,
    channel: str,
    state_flag: str,
    ifo: str,
    log_file: Path,
    verbose: bool,
):
    """Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file

    config = configparser.ConfigParser()
    section = "GW"
    config.add_section(section)

    config.set(section, "q-range", f"{q_min} {q_max}")
    config.set(section, "frequency-range", f"{f_min} {f_max}")
    config.set(section, "frametype", f"{ifo}_{frame_type}")
    config.set(section, "channels", f"{ifo}:{channel}")
    config.set(section, "cluster-dt", str(cluster_dt))
    config.set(section, "sample-frequency", str(sample_rate))
    config.set(section, "chunk-duration", str(chunk_duration))
    config.set(section, "segment-duration", str(segment_duration))
    config.set(section, "overlap-duration", str(overlap))
    config.set(section, "mismatch-max", str(mismatch_max))
    config.set(section, "snr-threshold", str(snr_thresh))
    # in an online setting, can also pass state-vector,
    # and bits to check for science mode
    config.set(section, "state-flag", f"{ifo}:{state_flag}")

    config_file_path = run_dir / f"omicron_{ifo}.ini"

    # write config file
    with open(config_file_path, "w") as configfile:
        config.write(configfile)

    # parse args into format expected by omicron
    omicron_args = [
        section,
        "--log-file",
        str(log_file),
        "--config-file",
        str(config_file_path),
        "--gps",
        f"{start}",
        f"{stop}",
        "--ifo",
        ifo,
        "-c",
        "request_disk=100M",
        "--output-dir",
        str(run_dir),
        "--skip-gzip",
        "--skip-rm",
    ]
    if verbose:
        omicron_args += ["--verbose"]

    # create and launch omicron dag
    omicron_main(omicron_args)

    # return variables necessary for glitch generation
    return ifo


def main(
    snr_thresh: float,
    start: int,
    stop: int,
    test_stop: int,
    q_min: float,
    q_max: float,
    f_min: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    window: float,
    datadir: Path,
    logdir: Path,
    channel: str,
    frame_type: str,
    sample_rate: float,
    state_flag: str,
    ifos: List[str],
    chunk_size: float = 4096,
    analyze_testing_set: bool = False,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates a set of glitches for both
        H1 and L1 that can be added to background

        First, an omicron job is launched via pyomicron
        (https://github.com/gwpy/pyomicron/). Next, triggers (i.e. glitches)
        above a given SNR threshold are selected, and data is queried
        for these triggers and saved in an h5 file.

    Arguments:
        snr_thresh: snr threshold above which to keep as glitch
        start: start gpstime
        stop: training stop gpstime
        test_stop: testing stop gpstime
        q_min: minimum q value of tiles for omicron
        q_max: maximum q value of tiles for omicron
        f_min: lowest frequency for omicron to consider
        cluster_dt: time window for omicron to cluster neighboring triggers
        chunk_duration: duration of data (seconds) for PSD estimation
        segment_duration: duration of data (seconds) for FFT
        overlap: overlap (seconds) between neighbouring segments and chunks
        mismatch_max: maximum distance between (Q, f) tiles
        window: half window around trigger time to query data for
        sample_rate: sampling frequency
        outdir: output directory to which signals will be written
        channel: channel name used to read data
        frame_type: frame type for data discovery w/ gwdatafind
        sample_rate: sampling frequency of timeseries data
        state_flag: identifier for which segments to use
        ifos: which ifos to generate glitches for
    """

    datadir = Path(datadir)
    datadir.mkdir(exist_ok=True, parents=True)
    glitch_file = datadir / "glitches.h5"

    if glitch_file.exists() and not force_generation:
        logging.info(
            "Glitch data already exists and forced generation is off. "
            "Not generating glitches"
        )
        return

    # nyquist
    f_max = sample_rate / 2

    glitches = {}
    snrs = {}
    times = {}
    train_futures = []

    run_dir = datadir / "omicron"
    train_run_dir = run_dir / "training"
    test_run_dir = run_dir / "testing"
    omicron_log_file = run_dir / "pyomicron.log"

    for ifo in ifos:
        train_ifo_dir = train_run_dir / ifo
        test_ifo_dir = test_run_dir / ifo
        train_ifo_dir.mkdir(exist_ok=True, parents=True)

        # launch pyomicron futures for training set and testing sets.
        # as train futures complete, launch glitch generation processes.
        # let test set jobs run in background as glitch data is queried
        pool = ThreadPoolExecutor(4)
        args = [
            q_min,
            q_max,
            f_min,
            f_max,
            sample_rate,
            cluster_dt,
            chunk_duration,
            segment_duration,
            overlap,
            mismatch_max,
            snr_thresh,
            frame_type,
            channel,
            state_flag,
            ifo,
            omicron_log_file,
            verbose,
        ]

        train_future = pool.submit(
            omicron_main_wrapper, start, stop, train_ifo_dir, *args
        )
        train_futures.append(train_future)

        if analyze_testing_set:
            test_ifo_dir.mkdir(exist_ok=True, parents=True)
            pool.submit(
                omicron_main_wrapper, stop, test_stop, test_ifo_dir, *args
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('intersections', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('outdir', help='Path to the Omicron output',
                        type=str)
    args = parser.parse_args()

    segments = np.load(args.intersections)
    for segment in segments:
        if os.path.exists(f'{args.outdir}/{segment[0]}_{segment[1]}/omicron/training/H1/triggers/H1:{CHANNEL}/') \
                and os.path.exists(f'{args.outdir}/{segment[0]}_{segment[1]}/omicron/training/L1/triggers/L1:{CHANNEL}/'):
            print(f'Skippping segment {segment[0]} {segment[1]} because it is already done')
            continue

        main(SNR_THRESH,
             segment[0],  # start
             segment[1],  # stop
             None,  # test_stop
             Q_MIN,
             Q_MAX,
             F_MIN,
             CLUSTER_DT,
             CHUNK_DURATION,
             SEGMENT_DURATION,
             OVERLAP,
             MISTMATCH_MAX,
             WINDOW,
             f'{args.outdir}/{segment[0]}_{segment[1]}/',
             None,  # logdir
             CHANNEL,
             FRAME_TYPE,
             GLITCH_SAMPLE_RATE,
             STATE_FLAG,
             IFOS)
        # wait a minute before submitting the next job
        time.sleep(60)
