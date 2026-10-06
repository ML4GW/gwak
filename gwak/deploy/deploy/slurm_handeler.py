import shutil
import logging
import subprocess
import time
from typing import Optional

from machinery import gwak_logger, Pathfinder
from machinery import (
    gwak_dir,
    gwak_output_dir,
    O4_bbc_short_0_data_dir,
    O4_bbc_short_1_data_dir
)
from deploy.libs.cluster_tools import write_slurm_config, write_export_config, write_infer_core_config
from infer_data import get_shifts_meta_data
from transforms import cohmode_to_dim

def slurm_infer_wrapper(
    slurm_batch: int,
    slurm_kwargs: dict,
    run_name: str,
    ifos: list[str],
    psd_length: float,
    Tb: int,
    stride_batch_size: int,
    sample_rate: int,
    data_format: str,
    shifts: list[float],
    project: str,
    image: Optional[str] = None,
    grpc_port: int = 8001,
    fname: Optional[Pathfinder] = None,
    ana_data: Optional[str] = None,
    model_repo_dir: Optional[Pathfinder] = None,
    result_dir: Optional[Pathfinder] = None,
    patients: int = 30,
    job_rate_limit: int = 1,
    inference_rate: float = 2,
    singularity_path: Optional[str] = None,
    ana_ver: str="O4b_cat1",
    data_ver: str="O4b_cat12",
    cl_config: str='S4_SimCLR_multiSignalAndBkg',
    coh_mode: str="real",
    fm_config: str='NF_onlyBkg',

    **kwargs,
):
    """ Timeslide and Hermes(Triton) handeler to generate test result for GWAK model on Slurm.
    Each Slurm job exports its own model repo and spins up its own Triton server.

    Arguments:
        slurm_batch -- Number of Slurm jobs used to split the input files.
        slurm_kwargs -- #SBATCH arguments and the deploy commands to run in each job.
        ifos -- The detectors strain to read in.
        psd_length -- Seconds of data required to estimate the PSD.
        Tb -- The amount of time slide duration to run on. Unit (Second).
        stride_batch_size -- Number of kernels analysised by one PSD.
        sample_rate -- Amount of strain data in second frame of a single detector.
        data_format -- The file format to look up for in the fname.
        shifts -- The unit shift per timeslide apply to each detector.
        project -- The GWAK to look up to.
        image -- Triton image to look up to.

    Keyword Arguments:
        grpc_port -- Port of the first job, each following job shifts it by 3. (default: {8001})
        fname -- Directory that stores the strain data to produce time slides.
        model_repo_dir -- Resolve to the export dir of each Slurm job if equals to None. (default: {None})
        result_dir -- Automatic resolve to gwak/gwak/output/infer if equals to None (default: {None})
        patients -- Max seconds to wait for Triton server to come online. (default: {30})
        job_rate_limit -- Maximum number of inference workers inside each Slurm job.
        inference_rate -- Numbers of kernel to run in one second. (default: {2})
        singularity_path -- Apptainer/Singularity binary used to run the Triton image. (default: {None})
    """


    Tb = int(Tb)
    grpc_port = int(grpc_port)
    output_dir = gwak_output_dir()
    deploy_dir = gwak_dir(suffix="gwak/deploy")()
    export_config_file = deploy_dir / "configs/export.yaml"

    ifo_str = ''.join(ifo[0] for ifo in ifos)
    prefix = f"{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
    dim_split = [
        6, # Embedding dimension
        cohmode_to_dim(coh_mode),
        1
    ]
    # File handling
    if result_dir is None:
        result_dir = output_dir(
            append_path=f"infer/{ifo_str}/{ana_ver}/{prefix}/{run_name}"
        )
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Define fname
    fname = fname(append_path=f"{ifo_str}/{ana_data}")
    if run_name == "bbc-short-0":
        fname = O4_bbc_short_0_data_dir(suffix=ifo_str)()
    if run_name == "bbc-short-1":
        fname = O4_bbc_short_1_data_dir(suffix=ifo_str)()

    log_file = result_dir / "log.log"
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

    if slurm_batch > len(fnames):
        slurm_batch = len(fnames)
    files_per_node = int(len(fnames)/slurm_batch)
    extra = len(fnames)%slurm_batch
    kernel_size = int(sample_rate * stride_batch_size / inference_rate)

    idx = 0
    width = len(str(slurm_batch))
    node_job_dir, node_fnames, node_segments = [], [], []
    for i in range(slurm_batch):
        job_dir = result_dir / f"Node_{i:0{width}d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        node_job_dir.append(job_dir)
        size = files_per_node + (1 if i < extra else 0)
        node_fnames.append(fnames[idx:idx+size])
        node_segments.append(segments[idx:idx+size])
        idx += size

    # Slurm job submission
    for node in range(slurm_batch):

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
            export_job_dir / f"{ifo_str}/{prefix}/{project}"
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
            image=image,
            grpc_port=grpc_port,
            patients=patients,
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

        node_slurm_kwargs = dict(slurm_kwargs)
        node_slurm_kwargs["job-name"] = f"{node:0{width}d}_{run_name}_TS"
        node_slurm_kwargs["output"] = job_dir / "output.log"
        node_slurm_kwargs["error"] = job_dir / "error.log"

        slurm_file = write_slurm_config(
            kwargs=node_slurm_kwargs,
            job_dir=job_dir,
            export_config=export_config,
            infer_config=infer_config,
        )
        logging.info(f" ")
        logging.info(f"Submitting {slurm_file}")
        logging.info(f" ")
        subprocess.run(["sbatch", f"{slurm_file}"])
        # Nodes may share a host, shift the ports used by Triton (grpc/http/metrics)
        grpc_port += 3

    logging.info(
        f"Infer result at: {result_dir}/inference_result"
    )
