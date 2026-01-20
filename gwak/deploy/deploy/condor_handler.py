import yaml
import shutil
import subprocess
from pathlib import Path
from infer_data import get_shifts_meta_data
from deploy.libs.cluster_tools import write_condor_config, write_export_config, write_infer_core_config
from typing import Optional
import os
import time
import h5py
import logging
import shutil
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

def condor_infer_wrapper(
    condor_nodes: int,
    condor_kwargs: dict,
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
    monitor_patients: int=3,
    job_rate_limit: int = 1,
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
        condor_nodes -- Max number of paralle running jobs. (default: {20})
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

    shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.log"
    triton_log = result_dir / "triton.log"
    gwak_logger(log_file)

    # Sequence preperation
    logging.info(f"Estimating required time slide to apply.")
    num_shifts, fnames, segments = get_shifts_meta_data(fname, Tb, shifts, data_format)
    fnames = [str(p) for p in fnames]
    files_per_node = int(len(fnames)/condor_nodes)
    if files_per_node == 0:
        files_per_node = 1
    # breakpoint()
    kernel_size = int(sample_rate * stride_batch_size / inference_sampling_rate)
    
    # Triton server setup
    ip = get_ip_address()
    # ip = "10.14.0.160"
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
        # monitor_patients = 15
        logging.info(f"Waiting {monitor_patients} seconds to recieve connetion to port {grpc_port}!")
        time.sleep(monitor_patients)
        # print("Model Loaded")
        # monitor = ServerMonitor(
        #     model_name=gwak_streamer,
        #     ips="localhost",
        #     filename=result_dir / f"server-stats-{stride_batch_size}.csv",
        #     grpc_port=grpc_port,
        #     model_version=-1,
        #     name="monitor",
        #     max_request_rate=1,
        # )
        # with monitor:

        sub_files = []
        # breakpoint()
        width = len(str(condor_nodes))
        start = time.time()
        for node in range(condor_nodes):

            files = fnames[node*files_per_node:(node+1)*files_per_node]
            # job_dir = output_dir / f"Slurm_Jobs/{prefix}/{run_name}" / f"Node_{node:02d}"
            job_dir = result_dir / f"Node_{node:0{width}d}"
            job_dir.mkdir(parents=True, exist_ok=True)
            # print(job_dir, flush=True)

            infer_config = write_infer_core_config(
                deploy_dir="deploy_dir",
                output_dir="output_dir",
                job_dir=job_dir,
                result_dir=result_dir,
                project=project,
                ip=ip,
                model_repo_dir=model_repo_dir or export_job_dir / "models", # Can be null, cause we export at node level=# model_repo_dir, # Can be null, cause we export at node level.
                image=image,
                grpc_port=grpc_port,
                patients=15,
                ifos=ifos,
                fnames=files,
                num_shifts=num_shifts,
                data_format=data_format,
                segments=segments[node*files_per_node:(node+1)*files_per_node],
                shifts=shifts,
                Tb=Tb,
                psd_length=psd_length,
                stride_batch_size=stride_batch_size,
                kernel_size=kernel_size,
                sample_rate=sample_rate,
                inference_sampling_rate=int(inference_sampling_rate),
                job_rate_limit=job_rate_limit,
            )

            condor_subs = write_condor_config(
                condor_kwargs=condor_kwargs,
                job_dir=job_dir,
                arguments=f"{file_path.parent}/cli.py condor_client",
                config=infer_config
            )
            sub_files.append(condor_subs)

        condor_submit_with_rate_limit(
            sub_files=sub_files,
            rate_limit=condor_nodes
        )

        # time.sleep(monitor_patients)
        logging.info(f"Time spent for inference: {(time.time() - start)/60:.02f}mins")
