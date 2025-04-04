import numpy as np
import h5py
import subprocess
from typing import Optional

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from background_utils import (
    get_conincident_segs,
    get_background,
    create_lcs,
    omicron_bashes,
    glitch_merger
)


def run_bash(bash_file):

    subprocess.run(
        ["bash", f"{bash_file}"], 
    )


def gwak_background(
    ifos: list[str], 
    state_flag: list[str], 
    channels: list[str], 
    frame_type: list[str], 
    ana_start: int,
    ana_end: int, 
    sample_rate: int, 
    save_dir: Path, 
    segments: str = None, # provide segments instead of start and end time
    # Omicron process
    omi_paras: Optional[dict] = None,
    **kwargs
):

    save_dir.mkdir(parents=True, exist_ok=True)

    if segments:

        segs = np.load(segments)

    else:

        segs = get_conincident_segs(
            ifos=ifos,
            start=ana_start,
            stop=ana_end,
            state_flag=state_flag,
        )

    for seg_num, (seg_start, seg_end) in enumerate(segs):

        print(f'Downloading segment from {seg_start} to {seg_end}')

        strains = get_background(
            seg_start=seg_start,
            seg_end=seg_end,
            ifos=ifos,
            channels=channels,
            frame_type=frame_type,
            sample_rate=sample_rate,
        )

        seg_dur = seg_end-seg_start
        file_name = f"background-{int(seg_start)}-{int(seg_dur)}.h5"

        with h5py.File(save_dir / file_name, "w") as g:

            for ifo in ifos:
                g.create_dataset(ifo, data=strains[ifo])
