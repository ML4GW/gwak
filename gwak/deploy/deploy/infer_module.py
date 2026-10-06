import time
import logging

from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from hermes.aeriel.serve import serve
from hermes.aeriel.monitor import ServerMonitor

from machinery import gwak_logger, gwak_dir
from deploy.libs.infer_utils import get_ip_address
from deploy.libs.infer_core import client_action, run_bash


def infer(
    # Triron and model
    project: str,
    model_repo_dir: Path, # >>> Dicided after slurm_batch >>> job_dir
    image: str,
    patients: int,
    # Data setting
    fnames: list, # >>> Dicided after slurm_batch (Data Cut)
    segments: list, # >>> Dicided after slurm_batch (Data Cut)
    num_shifts: int, # >>> Dicided after slurm_batch (Tb)
    shifts: list,
    Tb: int,
    job_dir: Path,
    result_dir: Path,
    # ip
    grpc_port: int, # >>> Dicided after slurm_batch
    data_format: str,
    psd_length: float,
    stride_batch_size: int,
    ifos: list,
    kernel_size: int,
    sample_rate: int,
    inference_sampling_rate: int,
    dim_split: list,
    job_rate_limit: int,
    singularity_path: Optional[str]=None,
    **kwargs
):


    log_file = job_dir / "log.log"
    triton_log = job_dir / "triton.log"
    gwak_logger(log_file)
    deploy_dir = gwak_dir(suffix="gwak/deploy")()
    # Triton server setup
    ip = get_ip_address()
    gwak_streamer = f"gwak-{project}-streamer"
    serve_context = serve(
        model_repo_dir,
        image,
        grpc_port,
        log_file=triton_log,
        wait=False,
        singularity_path=singularity_path
    )

    # The Triton excution to run
    arguments = deploy_dir / "deploy/triton_excution.py"
    # Spin up Triton Serve
    with serve_context as instance:

        # The wait argument will break in the version for this hermis version, 
        # so we build a wapper to handle it. 
        # We need to switch back to the wait argument once the issue is resolved.
        logging.info(f"Waiting up to {patients} seconds for Triton on port {grpc_port}!")
        instance.wait(endpoint=f"localhost:{grpc_port}", timeout=patients)

        monitor = ServerMonitor(
            model_name=gwak_streamer,
            ips="localhost",
            filename=job_dir / f"server-stats-{stride_batch_size}.csv",
            grpc_port=grpc_port,
            model_version=-1,
            name="monitor",
            max_request_rate=1,
        )
        # Wait 5 seconds for the monitor to spin up.
        # time.sleep(monitor_patients)
        time.sleep(5)
        with monitor:

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
                dim_split=dim_split,
                arguments=arguments,
            )

            with ThreadPoolExecutor(max_workers=job_rate_limit) as e:

                futures = [e.submit(run_bash, f) for f in bash_cmds]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Thread crashed: {e}")

            print(f"Time spent for inference: {(time.time() - start)/3600:.02f} hrs")
