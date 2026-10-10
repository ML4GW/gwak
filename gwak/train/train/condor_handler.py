import os
import time
import shutil
import logging

from pathlib import Path
from typing import Optional

from access.osdf_io import pelican_read_latest_project_image

from machinery import (
    gwak_logger,
    gwak_output_dir,
    gwak_container_output_dir,
)
from machinery.cluster_tools import (
    submit_condor_job,
    write_trainer_bash_file,
    write_container_condor_sub,
    condor_submit_with_rate_limit,
)


def condor_train_wrapper(
    ifo_mode: str,
    data_ver: str,
    data_tag: str,
    cl_config: str,
    coh_mode: str,
    fm_config: str,
    condor_kwargs: dict,
    project: str = "train",
    image_version: Optional[str] = None,
    wandb_mode: str = "offline",
    max_concurrent_node: int = 1,
):
    """ Submit the snakemake rules as a container universe Condor
    job and wait for the trained models to be transferred back.

    Arguments:
        ifo_mode -- Detector combination, e.g. HL.
        data_ver -- Data version wildcard, e.g. O4b_cat1-chunked.
        data_tag -- Dataset directory of data_ver, e.g. O4_MDC_background-chunked.

    Keyword Arguments:
        osdf_data_root -- OSDF collection holding {ifo_mode}/{data_tag},
            resolves to osdf:///igwn/cit/staging/$USER/Data/GWAK if None. (default: {None})
    """

    sub_files = []
    username = os.environ.get("USER")
    prefix = f"{ifo_mode}/{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
    osdf_data_root = f"osdf:///igwn/cit/staging/{username}/Data/GWAK/{ifo_mode}/{data_tag}"
    osdf_image_root = f"osdf:///igwn/cit/staging/{username}/Container/GWAK"
    job_dir = gwak_container_output_dir(suffix=f"condor/train/{prefix}")()
    initialdir = gwak_container_output_dir()()
    job_dir.mkdir(parents=True, exist_ok=True)
    gwak_logger(log_file=job_dir / "job.log")

    bash_file = write_trainer_bash_file(
        job_dir=job_dir,
        ifo_mode=ifo_mode,
        data_tag=data_tag,
        prefix=prefix,
        osdf_data_root=osdf_data_root,
        wandb_mode=wandb_mode,
        num_cores=condor_kwargs.get("request_cpus", 4),
    )

    condor_subs = write_container_condor_sub(
        job_dir=job_dir,
        image=pelican_read_latest_project_image(project_name="train"),
        condor_kwargs=condor_kwargs,
        initialdir=initialdir,
        executable=bash_file,
    )
    logging.info(f"Submitting Condor job with submit file:")
    logging.info(f"    {condor_subs}")
    submit_condor_job(condor_subs)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(condor_train_wrapper, as_positional=False)
