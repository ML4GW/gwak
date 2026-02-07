import os
import time
import logging

from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from hermes.aeriel.serve import serve
from hermes.aeriel.monitor import ServerMonitor

from deploy.libs import gwak_logger, Pathfinder, gwak_dir
from deploy.libs.infer_utils import get_ip_address
from deploy.libs.infer_core import client_action, run_bash


def infer(
    run_name: str,
    job_dir: Path,
    result_dir: Path,
    # Triron and model
    project: str,
    ip: str,
    grpc_port: int, # >>> Dicided after slurm_batc
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
    condor_fnames = []
    log_file = job_dir / "log.log"
    gwak_streamer = f"gwak-{project}-streamer"
    deploy_dir = gwak_dir(suffix="gwak/deploy")()
    condor_local = Path(os.environ["_CONDOR_SCRATCH_DIR"])
    gwak_logger(log_file)


    # Triton server setup
    for file in fnames:
        condor_fnames.append(condor_local / file)
    arguments = deploy_dir / "deploy/triton_excution.py"
    start = time.time()
    bash_cmds = client_action(
        fnames=condor_fnames,
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

    with ThreadPoolExecutor(max_workers=job_rate_limit) as e:

        futures = [e.submit(run_bash, f) for f in bash_cmds]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Thread crashed: {e}")

    hrs, rem = divmod(time.time() - start, 3600)
    mins, _ = divmod(rem, 60)
    print(f"Time spent for inference: {int(hrs)} hrs {int(mins)} mins")
