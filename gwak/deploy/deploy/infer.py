import os
import time
import h5py
import logging
import subprocess

import numpy as np

from tqdm import tqdm
from zlib import adler32
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from hermes.aeriel.serve import serve
from hermes.aeriel.monitor import ServerMonitor

from deploy.libs.infer_utils import get_ip_address 
from deploy.libs import gwak_logger
from deploy.libs.cluster_tools import make_infer_config, make_subfile, condor_submit_with_rate_limit
from infer_data import get_shifts_meta_data

EXTREME_CCSN = [
    "Pan_2021/FR",
    "Powell_2020/y20"
]


def bash_commnad_files(bash_file, command):
    bash_file = bash_file / "triton.sh"
    with open(bash_file, "w") as f:
        f.write(command)

    return bash_file


def run_bash(bash_file):

    subprocess.run(
        ["bash", f"{bash_file}"],  
    )

def infer(
    ifos: list[str],
    psd_length: float,
    Tb: int,
    stride_batch_size: int,
    sample_rate: int,
    fname: Path, 
    data_format: str,
    ccsn_repo: Path,
    shifts: list[float], 
    project: str,
    image: str,
    grpc_port: int = 8001,
    model_repo_dir: Optional[Path] = None,
    result_dir: Optional[Path] = None,
    job_rate_limit: int=20,
    monitor_patients: int=3,
    inference_sampling_rate: float = 2,
    inj_type: Optional[str]=None,
    cl_config: str='S4_SimCLR_multiSignalAndBkg',
    fm_config: str='NF_onlyBkg',
    **kwargs,
):
    """ Timeslide and Hermes(Triton) handeler to generate test result for GWAK model. 

    Arguments:
        ifos -- The detectors strain to read in. 
        psd_length -- Seconds of data required to estimate the PSD. 
        Tb -- The amount of time slide duration to run on. Unit (Second). 
        stride_batch_size -- Number of kernels analysised by one PSD.
        sample_rate -- Amount of strain data in second frame of a single detector. 
        fname -- Directory that stores the strain data to produce time slides. 
        data_format -- The file format to look up for in the fname. 
        ccsn_repo -- The path to look up simulated CCSN.
        shifts -- The unit shift per timeslide apply to each detector. 
        project -- The GWAK to look up to.
        image -- Triton image to look up to.

    Keyword Arguments:
        model_repo_dir -- Automatic resolve to gwak/gwak/output/export if equals to None. (default: {None})
        result_dir -- Automatic resolve to gwak/gwak/output/infer if equals to None (default: {None})
        job_rate_limit -- Max number of paralle running jobs. (default: {20})
        monitor_patients -- The addtional waiting time wating for monitor to return messages. (default: {3})
        inference_sampling_rate -- Numbers of kernel to run in one second. (default: {2})
        inj_type -- Class of wavform to inject on timeslide. (default: {None})
    """
    result_dir = None
    # File handling 
    file_path = Path(__file__).resolve()
    ifo_str = ''.join(ifo[0] for ifo in ifos)

    prefix = f"{cl_config}_{fm_config}_{ifo_str}"
    if model_repo_dir is None: 
        model_repo_dir = file_path.parents[2] / "output/export"
    model_repo_dir = model_repo_dir / prefix / project
    if result_dir is None: 
        result_dir = file_path.parents[2] / "output/infer"

    result_dir = result_dir / project / prefix # already passed through snakemake
    result_dir = result_dir.resolve()

    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.log"
    triton_log = result_dir / "triton.log"
    gwak_logger(log_file)

    # Sequence preperation
    logging.info(f"Estimating required time slide to apply.")
    num_shifts, fnames, segments = get_shifts_meta_data(fname, Tb, shifts, data_format)
    kernel_size = int(sample_rate * stride_batch_size / inference_sampling_rate)
    
    # Triton server setup
    ip = get_ip_address()
    gwak_streamer = f"gwak-{project}-streamer"
    serve_context = serve(
        model_repo_dir, 
        image, 
        grpc_port, 
        log_file=triton_log, 
        # wait=True
        wait=False,
        singularity_path="/apps/system/software/apptainer/latest/bin/singularity"
    )
    # The Triton excution to run
    arguments=Path("deploy/triton_excution.py").resolve()
    if inj_type is not None:
        arguments=Path("deploy/triton_inj_excution.py").resolve()
        os.environ["CCSN_FILE"] = str(Path("deploy/config/ccsn.yaml").resolve())
    # arguments=Path("deploy/triton_inj_excution.py").resolve()

    with serve_context:
        monitor_patients = 5
        logging.info(f"Waiting {monitor_patients} seconds to recieve connetion to port {grpc_port}!")
        time.sleep(monitor_patients)
        # monitor = ServerMonitor(
        #     model_name=gwak_streamer,
        #     ips="localhost",
        #     filename=result_dir / f"server-stats-{stride_batch_size}.csv",
        #     model_version=-1,
        #     name="monitor",
        #     max_request_rate=1,
        # )
        # with monitor:
        submit_num = 0
        bash_files = []
        sub_files = []

        start = time.time()
        for fname, (seg_start, seg_end) in zip(fnames, segments):
            for shift in range(num_shifts):
                # print(fname, shift)
                fingerprint = f"{seg_start}{seg_end}{shift}".encode()
                sequence_id = adler32(fingerprint)
                _shifts = [s * (shift + 1) for s in shifts]
                if Tb == 0: 
                    _shifts = [0, 0]
                job_dir = result_dir / f"condor/job_{submit_num:08d}" # Make this to flexable
                logging.info(f"Creating config at {job_dir}.")
                job_dir.mkdir(parents=True, exist_ok=True)
                config_file = make_infer_config(
                    job_dir=job_dir,
                    triton_server_ip=ip,
                    grpc_port=grpc_port,
                    gwak_streamer=gwak_streamer,
                    sequence_id=sequence_id,
                    strain_file=fname, 
                    data_format=data_format,
                    shifts=_shifts,
                    psd_length=psd_length,
                    # batch_size=batch_size,
                    stride_batch_size=stride_batch_size,
                    ifos=ifos,
                    kernel_size=kernel_size,
                    sample_rate=sample_rate,
                    inference_sampling_rate=inference_sampling_rate,
                    # inj_type=inj_type,
                )

        #         # Condor inference
        #         submit_file = make_subfile(
        #             job_dir=job_dir,
        #             arguments=arguments,
        #             config=config_file.resolve()
        #         )
        #         sub_files.append(submit_file)

        #         submit_num += 1
                
        # # Condor inference
        # condor_submit_with_rate_limit(
        #     sub_files=sub_files,
        #     rate_limit=job_rate_limit
        # )


                # Local inference
                cmd = f"python {str(arguments)} --config {config_file}"
                bash_files.append(bash_commnad_files(job_dir, cmd))

                submit_num += 1

        # Local inference
        with ThreadPoolExecutor(max_workers=job_rate_limit) as e:
        
            # for bash_file in bash_files:
            #     logging.info(f"Excuting {bash_file}")
            #     e.submit(run_bash, bash_file)
            futures = [e.submit(run_bash, f) for f in bash_files]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Thread crashed: {e}")


        logging.info(f"Time spent for inference: {(time.time() - start)/60:.02f}mins")
