from machinery.find_gpus import gpu_selector
import os 

import time
import shutil
import logging

# from access.osdf_io import pelican_read_latest_project_image

from typing import Optional
from contextlib import nullcontext
from hermes.aeriel.serve import serve
from hermes.aeriel.monitor import ServerMonitor

from infer_modules import get_ip_address 
from machinery import gwak_logger, Pathfinder
from machinery import (
    gwak_dir,
    gwak_container_output_dir,
    gwak_output_dir,
    O4_bbc_short_0_data_dir,
    O4_bbc_short_1_data_dir
)
from machinery.cluster_tools import (
    submit_condor_job,
    write_bash_file,
    write_condor_config,
    write_export_config,
    write_infer_core_config,
    condor_submit_with_rate_limit,
    write_container_condor_sub
)
from infer_data import get_shifts_meta_data
from transforms import cohmode_to_dim

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
    grpc_port: int = 8001,
    image: Optional[Pathfinder] = None,
    fname: Optional[Pathfinder] = None, 
    ana_data: Optional[str] = None,
    model_repo_dir: Optional[Pathfinder] = None,
    result_dir: Optional[Pathfinder] = None,
    server_patients: int=3, 
    monitor_patients: Optional[int]=3,
    max_concurrent_node: int = 8,
    job_rate_limit: int = 1,
    inference_rate: float = 2,
    inj_type: Optional[str]=None,
    singularity_path: Optional[str]=None,
    ana_ver: str="O4b_cat1",
    data_ver: str="O4b_cat12",
    cl_config: str='S4_SimCLR_multiSignalAndBkg',
    coh_mode: str="real",
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
        condor_nodes -- Number of Condor jobs used to split the input files.
        max_concurrent_node -- Maximum number of Condor jobs running at the same time.
        job_rate_limit -- Maximum number of inference workers inside each Condor job.
        monitor_patients -- The addtional waiting time wating for monitor to return messages. (default: {3})
        inference_sampling_rate -- Numbers of kernel to run in one second. (default: {2})
        inj_type -- Class of wavform to inject on timeslide. (default: {None})
    """
    patients = 5

    Tb = int(Tb)
    output_dir = gwak_output_dir()
    deploy_dir = gwak_dir(suffix="gwak/deploy")()
    export_config_file = deploy_dir / "configs/export.yaml"
    cli = deploy_dir / "deploy/cli.py"
    
    ifo_mode = ''.join(ifo[0] for ifo in ifos)
    prefix = f"{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
    username = os.environ.get("USER")
    osdf_data_root = f"osdf:///igwn/cit/staging/{username}/Data/GWAK/{ifo_mode}/{data_tag}"
    osdf_image_root = f"osdf:///igwn/cit/staging/{username}/Container/GWAK"

    dim_split = [
        6, # Embedding dimension
        cohmode_to_dim(coh_mode),
        1
    ]
    # File handling
    if model_repo_dir is None: 
        model_repo_dir = output_dir(
            append_path=f"export/{ifo_mode}/{prefix}/{project}"
        )
    if result_dir is None:
        result_dir = output_dir(
            append_path=f"infer/{ifo_mode}/{ana_ver}/{prefix}/{run_name}"
        )
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Define fname

    fname = fname(append_path=f"{ana_data}/{ifo_mode}")
    if run_name == "bbc-short-0":
        fname = O4_bbc_short_0_data_dir(suffix=ifo_mode)()
    if run_name == "bbc-short-1":
        fname = O4_bbc_short_1_data_dir(suffix=ifo_mode)()

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

    # HTCondor job submission
    for node in range(condor_nodes):

        job_dir = node_job_dir[node]
        export_job_dir = job_dir / "export"
        infer_job_dir = job_dir / "infer"
        export_job_dir.mkdir(parents=True, exist_ok=True)
        infer_job_dir.mkdir(parents=True, exist_ok=True)

        # Each node exports to its own repo to avoid racing on a shared one
        export_dir = {
            "class_path": "gwak_output_dir",
            "init_args": {"suffix": str(export_job_dir.relative_to(output_dir()))},
        }
        node_model_repo_dir = model_repo_dir or (
            export_job_dir / f"{ifo_mode}/{prefix}/{project}"
        )

        export_config = write_export_config(
            export_config_file=export_config_file,
            export_job_dir=export_job_dir,
            project=project,
            export_dir=export_dir,
            stride_batch_size=stride_batch_size,
            ifos=ifos,
            psd_length=psd_length,
            sample_rate=sample_rate,
            inference_rate=inference_rate,
            data_ver=data_ver,
            cl_config=cl_config,
            coh_mode=coh_mode,
            fm_config=fm_config,
        )

        # Keys must match deploy.infer_module.infer()
        infer_config = write_infer_core_config(
            run_name=run_name,
            job_dir=infer_job_dir,
            result_dir=result_dir,
            project=project,
            model_repo_dir=node_model_repo_dir,
            image=image(),
            grpc_port=grpc_port,
            patients=server_patients,
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
            dim_split=dim_split,
            job_rate_limit=job_rate_limit,
            singularity_path=singularity_path,
            keep_fname_dir=True, # Slurm reads strain in place, no scratch copy
        )

        cmds = [
            f"python {cli} export --config {export_config}",
            # f"python {cli} infer --config {infer_config}"
        ]
        bash_file = write_bash_file(
            bash_root=node_job_dir[node],
            files=node_fnames[node],
            commands=cmds
        )

        condor_subs = write_condor_config(
            condor_kwargs=condor_kwargs,
            job_dir=node_job_dir[node],
            executable=bash_file,
            config=infer_config
        )
        logging.info(f"Submitting Condor job with submit file:")
        logging.info(f"    {condor_subs}")
        submit_condor_job(condor_subs)
        grpc_port += 3
