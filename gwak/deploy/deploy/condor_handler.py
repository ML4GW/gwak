from deploy.libs.find_gpus import gpu_selector
import os 

gpu_list = gpu_selector(free_mem=10000)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]["uuid"]

import time
import shutil
import logging

from typing import Optional
from contextlib import nullcontext
from hermes.aeriel.serve import serve
from hermes.aeriel.monitor import ServerMonitor

from deploy.libs import get_ip_address 
from deploy.libs import gwak_logger, Pathfinder
from deploy.libs import gwak_dir, gwak_output_dir, O4_bbc_short_0_data_dir, O4_bbc_short_1_data_dir
from deploy.libs.cluster_tools import write_bash_file, write_condor_config, write_infer_core_config, condor_submit_with_rate_limit
from infer_data import get_shifts_meta_data


def condor_infer_wrapper(
    condor_nodes: int,
    condor_kwargs: dict,
    run_name: str,
    ifos: list[str],
    psd_length: float,
    Tb: int,
    stride_batch_size: int,
    sample_rate: int,
    data_format: str,
    shifts: list[float], 
    project: str,
    image: str,
    grpc_port: int = 8001,
    fname: Optional[Pathfinder] = None, 
    model_repo_dir: Optional[Pathfinder] = None,
    result_dir: Optional[Pathfinder] = None,
    server_patients: int=3, 
    monitor_patients: Optional[int]=3,
    job_rate_limit: int = 1,
    inference_rate: float = 2,
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
    Tb = int(Tb)
    output_dir = gwak_output_dir()
    deploy_dir = gwak_dir(suffix="gwak/deploy")()
    
    ifo_str = ''.join(ifo[0] for ifo in ifos)
    prefix = f"{cl_config}_{fm_config}_{ifo_str}"
    # File handling     
    if model_repo_dir is None: 
        model_repo_dir = output_dir(
            append_path=f"export/{prefix}/{project}"
        )
    if result_dir is None:
        result_dir = output_dir(
            append_path=f"infer/{prefix}/{run_name}"
        )
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Define fname    
    if run_name == "bbc-short-0":
        fname = O4_bbc_short_0_data_dir()
    if run_name == "bbc-short-1":
        fname = O4_bbc_short_1_data_dir()

    if fname is not None:
        fname = fname(append_path=ifo_str)
    else:
        fname = output_dir(append_path=f"BBC_AnalysisReady_Cat12/{ifo_str}")

    log_file = result_dir / "log.log"
    triton_log = result_dir / "triton.log"
    gwak_logger(log_file)

    logging.info(f"")
    logging.info(f"Generating timeslide data from:")
    logging.info(f"    {fname}")
    logging.info(f"")

    # Sequence preperation
    logging.info(f"Estimating required time slide to apply.")
    num_shifts, fnames, segments = get_shifts_meta_data(
        fname, Tb, shifts, data_format
    )
    fnames = [str(p) for p in fnames]

    if condor_nodes > len(fnames):
        condor_nodes = len(fnames)
    files_per_node = int(len(fnames)/condor_nodes)
    extra = len(fnames)%condor_nodes
    kernel_size = int(sample_rate * stride_batch_size / inference_rate)

    idx = 0
    width = len(str(condor_nodes))
    node_job_dir, node_fnames, node_segments = [], [], []
    for i in range(condor_nodes):
        job_dir = result_dir / f"Node_{i:0{width}d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        node_job_dir.append(job_dir)
        size = files_per_node + (1 if i < extra else 0)
        node_fnames.append(fnames[idx:idx+size])
        node_segments.append(segments[idx:idx+size])
        idx += size

    # Triton server setup
    ip = get_ip_address()
    cli = deploy_dir / "deploy/cli.py"
    gwak_streamer = f"gwak-{project}-streamer"
    serve_context = serve(
        model_repo_dir,
        image,
        grpc_port,
        log_file=triton_log, 
        wait=False,
    )

    # The Triton excution to run
    with serve_context:
        logging.info(
            f"Waiting {server_patients} seconds "
            f"to receive connection to port {grpc_port}!"
        )
        logging.info(f"Triton logs saved at {triton_log}")
        time.sleep(server_patients)

        monitor = nullcontext()
        if monitor_patients:
            monitor = ServerMonitor(
                model_name=gwak_streamer,
                ips="localhost",
                filename=result_dir / f"server-stats.csv",
                grpc_port=grpc_port,
                model_version=-1,
                name="monitor",
                max_request_rate=1,
            )
            logging.info(
                f"Waiting {monitor_patients} seconds "
                f"to receive connection to monitor!"
            )
            time.sleep(monitor_patients)
        with monitor:
            start_time = time.time()
            sub_files = []
            for node in range(condor_nodes):

                infer_config = write_infer_core_config(
                    run_name=run_name,
                    job_dir=node_job_dir[node],
                    result_dir=result_dir,
                    project=project,
                    ip=ip,
                    grpc_port=grpc_port,
                    ifos=ifos,
                    fnames=node_fnames[node],
                    num_shifts=num_shifts,
                    data_format=data_format,
                    segments=node_segments[node],
                    shifts=shifts,
                    Tb=Tb,
                    psd_length=psd_length,
                    stride_batch_size=stride_batch_size,
                    kernel_size=kernel_size,
                    sample_rate=sample_rate,
                    inference_sampling_rate=int(inference_rate),
                    job_rate_limit=job_rate_limit,
                )

                cmd = f"python {cli} condor_client --config {infer_config}"
                bash_file = write_bash_file(
                    bash_root=node_job_dir[node],
                    files=node_fnames[node],
                    command=cmd
                )
                condor_subs = write_condor_config(
                    condor_kwargs=condor_kwargs,
                    job_dir=node_job_dir[node],
                    executable=bash_file,
                    config=infer_config
                )
                sub_files.append(condor_subs)

            condor_submit_with_rate_limit(
                sub_files=sub_files,
                rate_limit=condor_nodes
            )

            run_time = (time.time() - start_time)

    days, rem = divmod(run_time, 86400)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    logging.info(
        f"Infer result at: {result_dir}/inference_result"
    )
    logging.info(
        f"Time spent for inference: "
        f"{int(days)}--{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
    )

 
