import subprocess
from pathlib import Path
from infer_data import get_shifts_meta_data
from deploy.libs.cluster_tools import write_slurm_config, write_infer_core_config
from typing import Optional


def slurm_infer_wrapper(
    slurm_batch: int,
    slurm_kwargs: dict,
    ifos: list,
    Tb: int,
    fname: Path, 
    psd_length: float,
    stride_batch_size: int,
    sample_rate: int,
    project: str,
    grpc_port: int,
    patients: int,
    image: str,
    shifts: list[float],
    data_format: str,
    inference_sampling_rate: int,
    job_rate_limit: int,
    cl_config: str,
    fm_config: str,
    model_repo_dir: Optional[str]=None,
    **kwargs,
):

    # Local path settings
    deploy_dir = Path(__file__).resolve().parents[1]
    output_dir = deploy_dir.parents[0] / "output"
    ifo_str = ''.join(ifo[0] for ifo in ifos)
    prefix = f"{cl_config}_{fm_config}_{ifo_str}"
    
    num_shifts, fnames, segments = get_shifts_meta_data(fname, Tb, shifts, data_format)
    fnames = [str(p) for p in fnames]
    # breakpoint()
    kernel_size = int(sample_rate * stride_batch_size / inference_sampling_rate)
    files_per_batch = int(len(fnames)/slurm_batch)
    if files_per_batch == 0:
        files_per_batch = 1

    for batch in range(slurm_batch):
        
        job_dir = output_dir / "Slurm_Job" / f"Node_{batch:02d}"
        export_job_dir = job_dir / "export"
        infer_job_dir = job_dir / "infer"
        # import shutil
        # shutil.rmtree(job_dir.parent)
        export_job_dir.mkdir(parents=True, exist_ok=True)
        infer_job_dir.mkdir(parents=True, exist_ok=True)
        slurm_kwargs["job-name"] = f"Triton_job_{batch:02d}"
        slurm_kwargs["output"] = job_dir / f"output.log"
        slurm_kwargs["error"] = job_dir / f"error.log"
        grpc_port = int(grpc_port+3)
        # breakpoint()s
        # Make config file for infer_core() and return the path of the config
        infer_config = write_infer_core_config(
            deploy_dir=deploy_dir,
            output_dir=output_dir,
            job_dir=infer_job_dir,
            project=project,
            model_repo_dir=model_repo_dir or export_job_dir / prefix / project, # Can be null, cause we export at node level=# model_repo_dir, # Can be null, cause we export at node level.
            image=image,
            grpc_port=grpc_port,
            patients=patients,
            ifos=ifos,
            fnames=fnames[batch*files_per_batch:(batch+1)*files_per_batch],
            num_shifts=num_shifts,
            data_format=data_format,
            segments=segments[batch*files_per_batch:(batch+1)*files_per_batch],
            shifts=shifts,
            Tb=Tb,
            psd_length=psd_length,
            stride_batch_size=stride_batch_size,
            kernel_size=kernel_size,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            job_rate_limit=job_rate_limit
        )

        # It can just be job_dir / "submit.slurm"
        slurm_file = write_slurm_config(
            kwargs=slurm_kwargs,
            job_dir=job_dir,
            project="combination",
            infer_config=infer_config,
            cl_config=cl_config,
            fm_config=fm_config,
        )
        
        subprocess.run(["sbatch", f"{slurm_file}"])
        
        



