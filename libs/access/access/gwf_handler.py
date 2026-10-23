import os
import re

import time

import h5py
import logging
import multiprocessing as mp

import numpy as np

from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries, TimeSeriesList
from gwpy.segments import DataQualityDict


def frame_span(source):
    """
    GPS [start, end) covered by a frame file, parsed from its
    T050017 file name: <OBS>-<TAG>-<GPSSTART>-<DURATION>.gwf
    Works for local paths, file:// and osdf:// URLs.
    """
    stem = Path(urlparse(str(source)).path).stem
    _, gps_start, duration = stem.rsplit("-", 2)
    return float(gps_start), float(gps_start) + float(duration)


def _read_one_frame(source, name, start, end, nproc, file_format, verbose):
    # Runs in a spawned child process, see IsolatedFrameReader.
    return TimeSeries.read(
        source=source,
        name=name,
        start=start,
        end=end,
        nproc=nproc,
        format=file_format,
        verbose=verbose
    )


class IsolatedFrameReader:
    """
    Read frame files in a spawned child process.

    The frame libraries (lalframe/frameCPP) can segfault on some files,
    which kills the whole Python process and can't be caught by
    try/except. Reading in a child turns a segfault into a
    BrokenProcessPool exception, so we can report the bad file and retry.
    "spawn" (not fork) keeps the child from inheriting library state
    from the parent, which is itself a common cause of these crashes.
    """

    def __init__(self):
        self._pool = None

    def read(self, source, name, start, end, nproc, file_format, verbose):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn")
            )
        future = self._pool.submit(
            _read_one_frame, 
            source, name, start, end, nproc, file_format, verbose
        )
        try:
            return future.result()
        except BrokenProcessPool:
            # The child died (e.g. SIGSEGV); start a fresh one next time.
            self.close()
            raise RuntimeError(
                f"Frame reader crashed (likely a segfault) on {source}"
            )

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def gwf_safe_read(
    files,
    name,
    start,
    end,
    file_format,
    sample_rate,
    nproc=1, 
    max_attempts=5,
    retry_delay=5,
    verbose=True,
):
    """
    Read `name` between `start` and `end` from a list of frame files.

    Files are read one at a time, each cropped to the requested span,
    in an isolated child process with per-file retries. The pieces are
    then joined and resampled once, so no resampling edge effects are
    introduced at file boundaries.
    """
    if isinstance(files, (str, Path)):
        source = [source]

    # Keep one copy of each frame (datafind can return the same file
    # from several locations) and order them in time.
    frames = {}
    for file in files:
        
        frames.setdefault(Path(urlparse(str(file)).path).name, file)
    frames = sorted(frames.values(), key=frame_span)

    pieces = []
    with IsolatedFrameReader() as reader:
        for frame in frames:
            f_start, f_end = frame_span(frame)
            read_start, read_end = max(start, f_start), min(end, f_end)
            if read_start >= read_end:
                continue

            if verbose:
                logging.info(f"[{name}] reading {frame}")
                logging.info(f"({read_start} - {read_end})")

            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    pieces.append(reader.read(
                        frame, name, read_start, read_end, 
                        nproc, file_format, verbose
                    ))
                    break

                except Exception as exc:
                    last_error = exc
                    logging.info(
                        f"[{name}] read attempt "
                        f"{attempt}/{max_attempts} of {frame} failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                    if attempt < max_attempts:
                        time.sleep(retry_delay)
            else:
                msg = (f"Failed to read {name} from {frame} "
                    f"after {max_attempts} attempts")
                raise RuntimeError(msg) from last_error

    if not pieces:
        raise RuntimeError(
            f"No frame files cover {name} between GPS {start} and {end}"
        )

    try:
        strain = TimeSeriesList(*pieces).join(gap="raise")
    except ValueError as exc:
        raise RuntimeError(
            f"Gap in frame data for {name} between GPS {start} and {end}"
        ) from exc
    del pieces

    if strain.span[0] > start or strain.span[1] < end:
        raise RuntimeError(
            f"Frame data for {name} covers {tuple(strain.span)}, "
            f"requested ({start}, {end})"
        )

    strain = strain.resample(sample_rate)

    if not np.all(np.isfinite(strain.value)):
        raise RuntimeError(f"Strain {name} contains NaN or Inf")

    return strain.value


########################
### File level utils ###
########################

def get_conincident_segs(
    ifos:list,
    start:int,
    stop:int,
    state_flag:list,
    dqsegdb_url:str="https://segments.igwn.org",
):

    segs = []
    query_flag = []

    for i, ifo in enumerate(ifos):
        query_flag.append(f"{ifo}:{state_flag[i]}")

    flags = DataQualityDict.query_dqsegdb(
        query_flag,
        start,
        stop,
        on_error="raise", # warn
        host=dqsegdb_url,
    )

    try:
        active_table = flags.intersection().active.to_table()
    except ValueError:
        logging.info(f"No conincident segment for {ifos} between {start} and {stop} at {state_flag}.")
        return segs

    for contents in active_table:
        segs.append((contents["start"], contents["end"]))

    return segs
    

def get_background(
    seg_start: int,
    seg_end: int, 
    ifos:list,
    frame_type:list,
    channels:list,
    sample_rate:int,
    verbose:bool=True,
    host:str="datafind.ldas.cit:80"
): 
    urltype = "file"
    if host == "datafind.igwn.org":
        urltype = "osdf"
    strains = {}

    logging.info(f"Fetching data from {host}")
    logging.info(
        f"Collecting strain data from {seg_start} to {seg_end} at {channels}"
    )
    for num, ifo in enumerate(ifos):

        frametype=f"{ifo}_{frame_type[num]}" ### LIGO uses frametype WITH the IFO name
        if ifo.startswith("V"):
            frametype=f"{frame_type[num]}" ### VIRGO uses frametype WITHOUT the IFO name

        files = find_urls(
            site=f"{ifo[0]}",
            frametype=frametype,
            gpsstart=seg_start,
            gpsend=seg_end,
            urltype=urltype,
            host=host,
        )
        
        logging.info(f"Found {len(files)} files for {ifo}")
        if len(files) == 0:
            raise ValueError(f"No files found for {ifo} between {seg_start} and {seg_end}")

        logging.info(f"Reading strain data betweeen {seg_start} and {seg_end}.")

        strains[ifo] = gwf_safe_read(
            files=files, 
            name=f"{ifo}:{channels[num]}", 
            start=seg_start, 
            end=seg_end, 
            file_format="gwf",
            sample_rate=sample_rate,
            nproc=8, 
            verbose=verbose
        )
        logging.info(f"Strain data for {ifo} collected")
        logging.info("")
    strains['GPS_start'] = seg_start
    strains['GPS_stop'] = seg_end

    return strains

#######################
### Burst Benchmark ###
#######################

def find_files_with_ifo(directory, start_time, end_time, ifo):
    matching_files = []
    # Pattern to capture IFO, start time, duration
    pattern = re.compile(r".*-(\w\d)_BurstBenchmark-(\d+)-(\d+)\.gwf")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            file_ifo = match.group(1)
            file_start = int(match.group(2))
            duration = int(match.group(3))
            file_end = file_start + duration
            # Match requested IFO and time overlap
            if file_ifo == ifo and not (file_end <= start_time or file_start >= end_time):
                full_path = os.path.join(directory, filename)
                matching_files.append(full_path)

    return sorted(matching_files)

def get_injections(
    seg_start: int,
    seg_end: int,
    ifos:list,
    channels:list,
    sample_rate:int,
    injections_dir:str='/scratch/burst.benchmark/o4b-2/',
    verbose:bool=True
):

    strains = {}
    logging.info(
        f"Collecting strain data from {seg_start} to {seg_end} at {channels}"
    )

    for num, ifo in enumerate(ifos):
        files = find_files_with_ifo(injections_dir, seg_start, seg_end, ifo)
        logging.info(f"Found {len(files)} files for {ifo}")
        if len(files) == 0:
            raise ValueError(f"No files found for {ifo} between {seg_start} and {seg_end}")

        logging.info(seg_start, seg_end)

        strains[ifo] = gwf_safe_read(
            files=files,
            name=f"{ifo}:{channels[num]}",
            start=seg_start,
            end=seg_end,
            nproc=8,
            file_format="gwf",
            sample_rate=sample_rate,
            verbose=verbose
        )
        logging.info(f"Strain data for {ifo} collected")
        logging.info(strains[ifo].shape)
        logging.info()
    strains['GPS_start'] = seg_start
    strains['GPS_stop'] = seg_end

    return strains


