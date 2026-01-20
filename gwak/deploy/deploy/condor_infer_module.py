import time
import logging

from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from hermes.aeriel.serve import serve
from hermes.aeriel.monitor import ServerMonitor

from deploy.libs import gwak_logger
from deploy.libs.infer_utils import get_ip_address
from deploy.libs.infer_core import client_action, run_bash


def infer(
    deploy_dir: Path,
    output_dir: Path,
    job_dir: Path,
    result_dir: Path,
    # Triron and model
    project: str,
    ip: str,
    model_repo_dir: Path, # >>> Dicided after slurm_batch >>> job_dir
    image: str,
    grpc_port: int, # >>> Dicided after slurm_batch
    patients: int,
    # Data setting
    ifos: list,
    fnames: list, # >>> Dicided after slurm_batch (Data Cut)
    num_shifts: int, # >>> Dicided after slurm_batch (Tb)
    data_format: str,
    segments: list, # >>> Dicided after slurm_batch (Data Cut)
    shifts: list,
    Tb: int,
    psd_length: float,
    stride_batch_size: int,
    kernel_size: int,
    sample_rate: int,
    inference_sampling_rate: int,
    job_rate_limit: int,
    singularity_path: Optional[str]=None,
    **kwargs
):

    # inference_sampling_rate = int(inference_sampling_rate)
    log_file = job_dir / "log.log"
    triton_log = job_dir / "triton.log"
    gwak_logger(log_file)
    # Triton server setup
    # ip = get_ip_address()
    # ip = "10.14.0.160"
    gwak_streamer = f"gwak-{project}-streamer"

    # The Triton excution to run
    # arguments = deploy_dir / "deploy/triton_excution.py"
    arguments = "/home/hongyin.chen/anti_gravity/gwak/gwak/deploy/deploy/triton_excution.py"
    start = time.time()
    bash_cmds = client_action(
        fnames=fnames,
        segments=segments,
        num_shifts=num_shifts,
        shifts=shifts,
        Tb=Tb,
        job_dir=job_dir,
        result_dir=result_dir,
        ip=ip,
        grpc_port=grpc_port,
        gwak_streamer=gwak_streamer,
        data_format=data_format,
        psd_length=psd_length,
        stride_batch_size=stride_batch_size,
        ifos=ifos,
        kernel_size=kernel_size,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        arguments=arguments,
    )
    
    # print(bash_cmds)
    with ThreadPoolExecutor(max_workers=job_rate_limit) as e:

        futures = [e.submit(run_bash, f) for f in bash_cmds]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Thread crashed: {e}")

    print(f"Time spent for inference: {(time.time() - start)/3600:.02f} hrs")
