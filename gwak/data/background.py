import numpy as np
import h5py
import subprocess
from typing import Optional

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from background_utils import (
    get_conincident_segs,
    get_background,
    get_injections,
    create_lcs,
    omicron_bashes,
    glitch_merger
)


def run_bash(bash_file):

    subprocess.run(
        ["bash", f"{bash_file}"], 
        cwd="../../pyomicron", 
    )


def gwak_background(
    ifos: list[str], 
    channels: list[str],
    ana_start: int,
    ana_end: int,
    sample_rate: int,
    save_dir: Path,
    state_flag: list[str]=None,
    frame_type: list[str]=None,
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


        if not frame_type and not state_flag:
            strains = get_injections(
                seg_start=seg_start,
                seg_end=seg_end,
                ifos=ifos,
                channels=channels,
                sample_rate=sample_rate,
            )

        else:
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

            for dname, dset in strains.items():
                g.create_dataset(dname, data=dset)

        bash_files = [] # List of omicron commands to excute in background.  
        if omi_paras is not None:
            
            for ifo, frametype in zip(ifos, frame_type):

                create_lcs(
                    ifo=ifo,
                    frametype=f"{ifo}_{frametype}",
                    start_time=seg_start,
                    end_time=seg_end,
                    output_dir= Path(omi_paras["out_dir"]) / f"Segs_{int(seg_start)}_{int(seg_end)}", 
                    urltype="file"
                )

            bash_scripts = omicron_bashes(
                ifos= ifos,
                start_time=seg_start,
                end_time=seg_end,
                # project_dir= Path(omi_paras["out_dir"]) / f"Segs_{seg_num:05d}", ### Change this seg-start seg-end
                project_dir= Path(omi_paras["out_dir"]) / f"Segs_{int(seg_start)}_{int(seg_end)}", ### Change this seg-start seg-end
                # INI
                q_range= omi_paras["q_range"],
                frequency_range= omi_paras["frequency_range"],
                frame_type= frame_type,
                channels= channels,
                cluster_dt= omi_paras["cluster_dt"],
                sample_rate= sample_rate,
                chunk_duration= omi_paras["chunk_duration"],
                segment_duration= omi_paras["segment_duration"],
                overlap_duration= omi_paras["overlap_duration"],
                mismatch_max= omi_paras["mismatch_max"],
                snr_threshold= omi_paras["snr_threshold"],
            )

            for bash_script in bash_scripts:
                bash_files.append(bash_script)


            with ThreadPoolExecutor(max_workers=8) as e: # 8 workers
            
                for bash_file in bash_files:
                    e.submit(run_bash, bash_file)

                # Generate a glitch_info.h5 file that stores omicron informations 
                glitch_merger(
                    ifos=ifos,
                    omicron_path=omi_paras["out_dir"],
                    channels=channels
                )