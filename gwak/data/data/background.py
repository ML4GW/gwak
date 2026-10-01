import h5py
import time
import shutil
import logging
import subprocess
import numpy as np

from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from machinery import authentications, gwak_logger, gwak_dir
from access.segment_utils import (
    load_segments, select_segments, write_segment_list, find_local_segments
)
from access.gwf_handler import get_injections
from access.gwf_handler import get_conincident_segs, get_background
from access.omicron_handler import create_lcs, omicron_bashes, glitch_merger



def make_seg_list(
    ifos: list[str],
    ana_start: int,
    ana_end: int,
    resolved_segments: Path,
    logger: Path,
    state_flag: list[str] = None,
    segment_type: Optional[str] = None,
    dqsegdb_url: str = "https://segments.igwn.org",
    host: Optional[str] = None,
    version_tag: Optional[str] = None,
    **kwargs
):
    """
    Resolve the segments to download and save them to `segment_list`
    (.npy, or text with one "start end" per line). The segments come
    from, in order: seg_start/seg_end, `segments`, the local per-IFO
    lists in `segment_dir`, and finally a DQSegDB query of state_flag.

    Args:
        segment_type: Segment list name in segment_dir,
            e.g. original.o4b-2.
    """
    gwak_logger(logger)
    if host == "datafind.igwn.org":
        authentications.check_scitoken()

    segment_dir = gwak_dir(suffix="gwak/data/segments")()
    segs = find_local_segments(segment_dir, segment_type, ifos)
    logging.info(f"Using the local {segment_type} segment lists in {segment_dir}")

    if segs is None:
        segs = get_conincident_segs(
            ifos=ifos,
            start=ana_start,
            stop=ana_end,
            state_flag=state_flag,
            dqsegdb_url=dqsegdb_url,
        )

    write_segment_list(segs, resolved_segments)

    return segs


def gwak_background(
    ifos: list[str], 
    channels: list[str],
    sample_rate: int,
    save_dir: Path,
    logger: Path,
    verbose: Optional[bool] = False,
    version_tag: Optional[str] = None,
    host: str = "datafind.ldas.cit:80",
    state_flag: list[str]=None,
    frame_type: list[str]=None,
    segments: str = None, # provide segments instead of start and end time
    compression: Optional[str] = None,
    skip_background_generation: Optional[bool]=False,
    # Per-segment jobs (condor / slurm)
    seg_index: Optional[int] = None,
    overwrite: bool = False,
    **kwargs
):
    """
    Download the strain of the segments listed in `segments`
    (made by `make_seg_list`).

    Args:
        segments: Segment list (.npy/text) from `make_seg_list`.
        overwrite: Re-download segments whose output file exists.
    """
    gwak_logger(logger)
    # File handling
    ifo_abbrs = "".join(ifo[0] for ifo in ifos)
    save_dir.mkdir(parents=True, exist_ok=True)

    if host == "datafind.igwn.org":
        authentications.check_scitoken()

    segs = select_segments(load_segments(segments), seg_index)
    # Read data from local or osdf
    for seg_num, (seg_start, seg_end) in enumerate(segs):

        seg_dur = seg_end - seg_start
        file_name = f"background-{int(seg_start)}-{int(seg_dur)}.h5"
        if (save_dir / file_name).exists() and not overwrite:
            logging.info(f"{save_dir / file_name} exists, skipping")
            continue

        logging.info(f'Downloading segment from {seg_start} to {seg_dur}')    
        if not frame_type and not state_flag:
            strains = get_injections(
                seg_start=seg_start,
                seg_end=seg_end,
                ifos=ifos,
                channels=channels,
                sample_rate=sample_rate,
                verbose=verbose,
            )

        else:
            strains = get_background(
                seg_start=seg_start,
                seg_end=seg_end,
                ifos=ifos,
                channels=channels,
                frame_type=frame_type,
                sample_rate=sample_rate,
                verbose=verbose,
                host=host,
            )

        chunks = None
        if compression is not None:
            chunks = (4096*32,)

        # Write to a temporary file and rename, so a killed job never
        # leaves a partial file that would be skipped on resubmission.
        tmp_file = save_dir / f".{file_name}.tmp"
        with h5py.File(tmp_file, "w") as g:

            for dname, dset in strains.items():

                if dname in ["GPS_start", "GPS_stop"]:                
                    g.attrs[dname] = dset
                else:
                    g.create_dataset(
                        dname,
                        data=dset,
                        compression=compression,
                        chunks=chunks
                    )
        tmp_file.replace(save_dir / file_name)
        logging.info(f"Saved {save_dir / file_name}")
