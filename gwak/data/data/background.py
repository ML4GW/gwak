import h5py
import time
import shutil
import subprocess
import numpy as np

from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from background_utils import (
    get_conincident_segs,
    check_scitoken,
    get_background,
    get_injections,
    create_lcs,
    omicron_bashes,
    glitch_merger
)


FILE_PATH = Path(__file__).resolve()
PYOMICRON_PATH = FILE_PATH.parents[2] / "pyomicron"
OUTPUT_PATH = FILE_PATH.parents[1] / "output"

def run_omicron_bash_file(bash_file):

    subprocess.run(
        ["bash", f"{bash_file}"], 
        cwd=PYOMICRON_PATH, 
    )


def gwak_background(
    ifos: list[str], 
    channels: list[str],
    ana_start: int,
    ana_end: int,
    sample_rate: int,
    save_dir: Path,
    host: str = "datafind.ldas.cit:80",
    state_flag: list[str]=None,
    frame_type: list[str]=None,
    segments: str = None, # provide segments instead of start and end time
    compression: Optional[str] = None,
    skip_background_generation: Optional[bool]=False,
    # Omicron process
    omi_paras: Optional[dict] = None,
    **kwargs
):

    if host == "datafind.igwn.org":
        check_scitoken()

    # File handling
    ifo_abbrs = "".join(ifo[0] for ifo in ifos)
    save_dir.mkdir(parents=True, exist_ok=True)
    if omi_paras is not None:
        
        omicron_bash_files = [] # List of omicron commands to excute in background.     
        out_dir = omi_paras.get("out_dir") or (OUTPUT_PATH / f"omicron/{ifo_abbrs}")
        out_dir = Path(out_dir)
        if omi_paras["clear_out_dir"] and out_dir.exists():
            print(f"Cleaning up folder {out_dir}")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Find segments
    if segments:

        segs = np.load(segments)

    else:

        segs = get_conincident_segs(
            ifos=ifos,
            start=ana_start,
            stop=ana_end,
            state_flag=state_flag,
            host=host
        )

    # Run data fetching and Omicron process
    for seg_num, (seg_start, seg_end) in enumerate(segs):

        seg_dur = seg_end-seg_start
        if not skip_background_generation:

            print(f'Downloading segment from {seg_start} to {seg_dur}')    
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
                    host=host,
                )

            file_name = f"background-{int(seg_start)}-{int(seg_dur)}.h5"

            chunks = None
            if compression is not None:
                chunks = (4096*32,)
            with h5py.File(save_dir / file_name, "w") as g:

                for dname, dset in strains.items():
                    g.create_dataset(
                        dname, 
                        data=dset, 
                        compression=compression,
                        chunks=chunks
                    )

        # Handle Omicron processing
        if omi_paras is not None:

            for ifo, frametype in zip(ifos, frame_type):

                frametype_name = f"{ifo}_{frametype}"
                if ifo == "V1":
                    frametype_name = frametype
                create_lcs(
                    ifo=ifo,
                    frametype=frametype_name,
                    start_time=seg_start,
                    end_time=seg_end,
                    output_dir= Path(out_dir) / f"Segs_{int(seg_start)}_{int(seg_dur)}", 
                    urltype="file",
                    # host=host, # Let it use default CIT datafind server
                )

            bash_scripts = omicron_bashes(
                ifos=ifos,
                start_time=seg_start,
                end_time=seg_end,
                project_dir=Path(out_dir) / f"Segs_{int(seg_start)}_{int(seg_dur)}", 
                # INI
                q_range=omi_paras["q_range"],
                frequency_range=omi_paras["frequency_range"],
                frame_type=frame_type,
                channels=channels,
                cluster_dt=omi_paras["cluster_dt"],
                sample_rate=sample_rate,
                chunk_duration=omi_paras["chunk_duration"],
                segment_duration=omi_paras["segment_duration"],
                overlap_duration=omi_paras["overlap_duration"],
                mismatch_max=omi_paras["mismatch_max"],
                snr_threshold=omi_paras["snr_threshold"],
            )

            for bash_script in bash_scripts:
                if seg_dur <= 64:
                    continue
                omicron_bash_files.append(bash_script)

    if omi_paras is not None:

        bash_scripts = sorted(bash_scripts)

        print("Launching Omicron....")
        with ThreadPoolExecutor(max_workers=omi_paras["max_workers"]) as e:
            
            for conut, bash_file in enumerate(omicron_bash_files):
            
                e.submit(run_omicron_bash_file, bash_file.resolve())
                print(f"Run {bash_file}")
                time.sleep(0.1)
                print("====")

        # Generate a glitch_info.h5 file that stores omicron informations 
        for seg_num, (seg_start, seg_end) in enumerate(segs):
            seg_dur = seg_end-seg_start

            glitch_merger(
                ifos=ifos,
                omicron_path=out_dir / f"Segs_{int(seg_start)}_{int(seg_dur)}",
                channels=channels
            )