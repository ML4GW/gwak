import os
import time
import shutil
import logging

from pathlib import Path
from typing import Optional

from machinery import (
    gwak_logger,
    gwak_output_dir,
    gwak_container_output_dir,
)
from machinery.cluster_tools import (
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
    """ Submit the production_train_cl rule as a container universe Condor
    job and wait for the trained model to be transferred back.

    Arguments:
        ifo_mode -- Detector combination, e.g. HL.
        data_ver -- Data version wildcard, e.g. O4b_cat1-chunked.
        data_tag -- Dataset directory of data_ver, e.g. O4_MDC_background-chunked.
        cl_config -- Training config name under gwak/train/configs.
        condor_kwargs -- Extra/overriding submit file entries.

    Keyword Arguments:
        project -- Project image to run in. (default: {"train"})
        image -- Image file under osdf staging Container/GWAK,
            resolves to {project}.sif if None. (default: {None})
        job_dir -- Directory for the submit file, job logs and transferred
            Outputs. Resolves to gwak/output/condor/train/... if None. (default: {None})
        osdf_data_root -- OSDF collection holding {ifo_mode}/{data_tag},
            resolves to osdf:///igwn/cit/staging/$USER/Data/GWAK if None. (default: {None})
        wandb_mode -- WANDB_MODE inside the job. (default: {"offline"})
        max_concurrent_node -- Maximum number of Condor jobs running at the same time. (default: {1})
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
        image=f"{osdf_image_root}/{image_version}",
        condor_kwargs=condor_kwargs,
        initialdir=initialdir,
        executable=bash_file,
    )
    # breakpoint()
    # sub_files.append(condor_subs)

    # condor_submit_with_rate_limit(
    #     sub_files=sub_files,
    # )

    # run_time = (time.time() - start_time)
    # days, rem = divmod(run_time, 86400)
    # hrs, rem = divmod(rem, 3600)
    # mins, secs = divmod(rem, 60)
    # logging.info(
    #     f"Time spent for training: "
    #     f"{int(days)}--{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
    # )

    # model = job_dir / "Outputs/model_JIT.pt"
    # if not model.exists():
    #     raise RuntimeError(
    #         f"Condor training finished without {model}, "
    #         f"check {job_dir / 'job.err'} and {job_dir / 'job.log'}"
    #     )
    # logging.info(f"Trained model at: {model}")

    # return model


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(condor_train_wrapper, as_positional=False)
