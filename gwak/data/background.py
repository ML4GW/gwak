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
    # Omicron process
    omi_paras: Optional[dict] = None,
    **kwargs
):

    save_dir.mkdir(parents=True, exist_ok=True)
    segs = get_conincident_segs(
        ifos=ifos,
        start=ana_start,
        stop=ana_end,
        state_flag=state_flag, 
    )


    for seg_num, (seg_start, seg_end) in enumerate(segs):

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


        bash_files = [] # List of omicron commands to excute in background.  
        # if omi_paras is not None:
            
        #     for ifo, frametype in zip(ifos, frame_type):

        #         create_lcs(
        #             ifo=ifo,
        #             frametype=f"{ifo}_{frametype}",
        #             start_time=seg_start,
        #             end_time=seg_end,
        #             output_dir= Path(omi_paras["out_dir"]) / f"Segs_{seg_num:05d}", 
        #             urltype="file"
        #         )

        #     bash_scripts = omicron_bashes(
        #         ifos= ifos,
        #         start_time=seg_start,
        #         end_time=seg_end,
        #         project_dir= Path(omi_paras["out_dir"]) / f"Segs_{seg_num:05d}",
        #         # INI
        #         q_range= omi_paras["q_range"],
        #         frequency_range= omi_paras["frequency_range"],
        #         frame_type= frame_type,
        #         channels= channels,
        #         cluster_dt= omi_paras["cluster_dt"],
        #         sample_rate= sample_rate,
        #         chunk_duration= omi_paras["chunk_duration"],
        #         segment_duration= omi_paras["segment_duration"],
        #         overlap_duration= omi_paras["overlap_duration"],
        #         mismatch_max= omi_paras["mismatch_max"],
        #         snr_threshold= omi_paras["snr_threshold"],
        #     )

        #     for bash_script in bash_scripts:
        #         bash_files.append(bash_script)


        #     # with ThreadPoolExecutor(max_workers=8) as e: # 8 workers
            
        #     #     for bash_file in bash_files:
        #     #         e.submit(run_bash, bash_file)

        #     #     # Generate a glitch_info.h5 file that stores omicron informations 
        #     #     glitch_merger = glitch_merger(
        #     #         ifos=ifos,
        #     #         omicron_path=omi_paras["out_dir"],
        #     #         channels=channels
        #     #     )