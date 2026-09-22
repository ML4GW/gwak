import re
import os
import time
import yaml
import logging
import subprocess

from typing import Union
from pathlib import Path


def wait_for_file(path, timeout=6000, interval=1):
    start = time.time()

    while not os.path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Timed out waiting for {path}"
            )

        time.sleep(interval)


def write_bash_file(
    bash_root: Path,
    files: list,
    command: str
):
    bash_file = bash_root / "condor_cmd.sh"

    gwak_root = Path(os.getenv("GWAK_ROOT"))
    gwak_env = gwak_root / ".gwak/env.sh"
    deploy_env = gwak_root / "gwak/deploy/.venv/bin/activate"

    with bash_file.open("w") as sh_file:
        sh_file.write("#!/bin/bash\n")
        sh_file.write("set -euo pipefail\n\n")

        sh_file.write(
            'echo "===== GWAK Condor worker ====="\n'
        )
        sh_file.write(
            'echo "Hostname: $(hostname)"\n'
        )
        sh_file.write(
            'echo "PWD: $(pwd)"\n'
        )
        sh_file.write(
            'echo "_CONDOR_SCRATCH_DIR: '
            '${_CONDOR_SCRATCH_DIR:-undefined}"\n'
        )
        sh_file.write("\n")

        sh_file.write(
            'echo "HDF5_USE_FILE_LOCKING before: '
            '${HDF5_USE_FILE_LOCKING:-undefined}"\n'
        )

        sh_file.write(
            "export HDF5_USE_FILE_LOCKING=FALSE\n"
        )

        sh_file.write(
            'echo "HDF5_USE_FILE_LOCKING after: '
            '$HDF5_USE_FILE_LOCKING"\n'
        )

        sh_file.write("\n")

        #
        # Verify shared GWAK environment is visible.
        #
        sh_file.write(
            f'echo "Checking GWAK environment: {gwak_env}"\n'
        )

        sh_file.write(
            f'test -r "{gwak_env}" || '
            f'{{ echo "ERROR: Cannot read {gwak_env}" >&2; '
            f'exit 70; }}\n'
        )

        sh_file.write(
            f'echo "Checking deploy environment: {deploy_env}"\n'
        )

        sh_file.write(
            f'test -r "{deploy_env}" || '
            f'{{ echo "ERROR: Cannot read {deploy_env}" >&2; '
            f'exit 71; }}\n'
        )

        sh_file.write("\n")

        #
        # Stage the HDF5 files into local Condor scratch.
        #
        for file in files:
            sh_file.write(
                f'echo "Copying {file}"\n'
            )

            sh_file.write(
                f'cp "{file}" "$_CONDOR_SCRATCH_DIR/"\n'
            )

        sh_file.write("\n")

        sh_file.write(
            'echo "Files under $_CONDOR_SCRATCH_DIR:"\n'
        )

        sh_file.write(
            'ls -lh "$_CONDOR_SCRATCH_DIR"\n'
        )

        sh_file.write("\n")

        sh_file.write(
            f'source "{gwak_env}"\n'
        )

        sh_file.write(
            f'source "{deploy_env}"\n'
        )

        sh_file.write("\n")

        #
        # exec ensures the Python exit status becomes
        # the Condor job exit status.
        #
        sh_file.write(
            f"exec {command}\n"
        )

    bash_file.chmod(0o755)

    return bash_file


def write_export_config(
    **export_kwargs
):

    # Load arguments from the main export file.
    with open(
        export_kwargs["export_config_file"],
        "r"
    ) as yaml_file:

        export_args = yaml.safe_load(
            yaml_file
        )

    # Replace arguments for inference.
    for key, item in export_kwargs.items():

        if key in (
            "export_config_file",
            "export_job_dir"
        ):
            continue

        export_args[key] = item

    export_config = (
        Path(
            export_kwargs["export_job_dir"]
        )
        / "export.yaml"
    )

    with open(
        export_config,
        "w"
    ) as yaml_file:

        for key, value in export_args.items():

            if value is None:
                value = "null"

            yaml_file.write(
                f"{key}: {value}\n"
            )

    return export_config


def write_condor_config(
    condor_kwargs,
    job_dir,
    executable,
    config
):
    job_dir = Path(job_dir)
    executable = Path(executable)

    condor_config = {}

    submit_file = job_dir / "condor.sub"
    job_out = job_dir / "job.out"
    job_err = job_dir / "job.err"
    job_log = job_dir / "job.log"

    #
    # Create stdout/stderr files on the submit node.
    #
    job_out.touch(exist_ok=True)
    job_err.touch(exist_ok=True)

    #
    # Basic Condor configuration.
    #
    condor_config["universe"] = "vanilla"

    #
    # IMPORTANT:
    #
    # Do NOT execute the launcher directly from /home.
    #
    # Run /bin/bash on the execute node and transfer only
    # condor_cmd.sh into the Condor scratch directory.
    #
    condor_config["executable"] = "/bin/bash"
    condor_config["arguments"] = executable.name

    #
    # Condor event log remains on the submit node.
    #
    condor_config["log"] = job_log

    #
    # Stream stdout/stderr back to the submit node.
    #
    condor_config["output"] = job_out
    condor_config["error"] = job_err

    condor_config["stream_output"] = True
    condor_config["stream_error"] = True

    #
    # Explicitly enable file transfer for the small launcher.
    #
    # /bin/bash itself already exists on the execute node, so
    # do not transfer the executable.
    #
    condor_config["should_transfer_files"] = "YES"
    condor_config["transfer_executable"] = False

    #
    # Transfer only condor_cmd.sh.
    #
    # It will appear in $_CONDOR_SCRATCH_DIR using its basename.
    #
    condor_config["transfer_input_files"] = str(executable)

    #
    # IMPORTANT:
    #
    # The GWAK script copies many GB of HDF5 files into the
    # Condor scratch directory. We do NOT want Condor to copy
    # those scratch files back to the submit host.
    #
    condor_config["transfer_output_files"] = '""'
    condor_config["when_to_transfer_output"] = "ON_EXIT"

    #
    # Apply requested resources / accounting settings.
    #
    for key, value in condor_kwargs.items():
        condor_config[key] = value

    with submit_file.open("w") as f:
        for key, value in condor_config.items():
            f.write(f"{key} = {value}\n")

        f.write("queue\n")

    return submit_file


def write_slurm_config(
    kwargs,
    job_dir,
    export_config,
    infer_config,
):

    file_path = (
        Path(__file__).resolve()
    )

    filename = (
        job_dir
        / "submit.slurm"
    )

    deploy_app_path = (
        file_path.parents[2]
    )

    export_cmd = (
        f"{kwargs['deploy_cmd']['export'][0]} "
        f"--config {export_config} "
    )

    infer_cmd = (
        f"{kwargs['deploy_cmd']['infer'][0]} "
        f"--config {infer_config}"
    )

    kwargs["gres"] = (
        f"gpu:{kwargs['gpu_per_node']}"
    )

    if kwargs["gpu_card"] is not None:

        kwargs["gres"] = (
            f"gpu:"
            f"{kwargs['gpu_card']}:"
            f"{kwargs['gpu_per_node']}"
        )

    config_content = [
        "#!/bin/bash"
    ]

    for item, key in kwargs.items():

        if item in (
            "gpu_card",
            "gpu_per_node",
            "cmd",
            "deploy_cmd",
        ):
            continue

        config_content.append(
            f"#SBATCH --{item}={key}"
        )

    config_content.append("")

    config_content.append(
        f"cd {deploy_app_path}"
    )

    cmds = (
        kwargs.get("cmd")
        or []
    )

    for cmd in cmds:

        config_content.append(
            cmd
        )

    config_content.append(
        export_cmd
    )

    config_content.append(
        infer_cmd
    )

    config_content.append("")

    with open(
        filename,
        "w"
    ) as f:

        f.write(
            "\n".join(
                config_content
            )
        )

    print(
        f"SLURM script written to: "
        f"{filename}"
    )

    return filename


def write_infer_core_config(
    **infer_core_kwargs
):

    yaml_file = (
        Path(
            infer_core_kwargs[
                "job_dir"
            ]
        )
        / "config.yaml"
    )

    with open(
        yaml_file,
        "w"
    ) as f:

        for key, value in (
            infer_core_kwargs.items()
        ):

            if (
                key == "fnames"
                and isinstance(
                    value,
                    list
                )
            ):

                f.write(
                    f"{key}:\n"
                )

                for item in value:

                    f.write(
                        f"  - "
                        f"{Path(item).name}\n"
                    )

            elif (
                key == "segments"
                and isinstance(
                    value,
                    list
                )
            ):

                f.write(
                    f"{key}:\n"
                )

                for item in value:

                    f.write(
                        f"  - {item}\n"
                    )

            else:

                if value is None:
                    value = "null"

                f.write(
                    f"{key}: "
                    f"{value}\n"
                )

    return yaml_file


def write_infer_config(
    job_dir: Path,
    result_dir: Path,
    triton_server_ip,
    grpc_port,
    gwak_streamer,
    sequence_id,
    strain_file: Union[
        str,
        Path
    ],
    data_format: str,
    shifts: list,
    psd_length: float,
    stride_batch_size: int,
    ifos: list,
    kernel_size: int,
    dim_split: list,
    sample_rate=2048,
    inference_sampling_rate=1,
):

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    config_file = (
        job_dir
        / "config.yaml"
    )

    with open(
        config_file,
        "w"
    ) as f:

        for key, value in (
            locals().items()
        ):

            if key in (
                "config_file",
                "f"
            ):
                continue

            f.write(
                f"{key}: "
                f"{value}\n"
            )

    return config_file


def submit_condor_job(
    sub_file: Path
):

    result = subprocess.run(
        [
            "condor_submit",
            str(sub_file),
        ],
        cwd=sub_file.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"condor_submit failed for "
            f"{sub_file}\n"
            f"stdout:\n"
            f"{result.stdout}\n"
            f"stderr:\n"
            f"{result.stderr}"
        )

    match = re.search(
        r"submitted to cluster (\d+)",
        result.stdout
    )

    if not match:

        raise RuntimeError(
            "condor_submit returned "
            "success but no cluster ID "
            "was found for "
            f"{sub_file}\n"
            f"stdout:\n"
            f"{result.stdout}"
        )

    job_id = (
        match.group(1)
        + ".0"
    )

    logging.info(
        f"Job {job_id} "
        "submitted successfully!"
    )

    return job_id


def _query_condor_job(
    job_id: str
):
    """
    Query a job currently present
    in condor_q.

    Returns
    -------
    job_state, hold_reason

    job_state is None when the job
    has already left condor_q.
    """

    result = subprocess.run(
        [
            "condor_q",
            job_id,
            "-af",
            "JobStatus",
            "HoldReason",
        ],
        capture_output=True,
        text=True,
    )

    if "SECMAN" in result.stderr:

        logging.warning(
            "Temporary HTCondor "
            "SECMAN error while "
            f"checking {job_id}: "
            f"{result.stderr.strip()}"
        )

        return (
            "SECMAN",
            None
        )

    if result.returncode != 0:

        raise RuntimeError(
            f"condor_q failed for "
            f"{job_id}\n"
            f"stdout:\n"
            f"{result.stdout}\n"
            f"stderr:\n"
            f"{result.stderr}"
        )

    job_info = (
        result.stdout.strip()
    )

    if not job_info:

        return (
            None,
            None
        )

    parts = (
        job_info.split(
            maxsplit=1
        )
    )

    try:

        job_state = int(
            parts[0]
        )

    except ValueError as exc:

        raise RuntimeError(
            "Could not parse "
            f"JobStatus for "
            f"{job_id}: "
            f"{job_info}"
        ) from exc

    hold_reason = (
        parts[1]
        if len(parts) > 1
        else "No HoldReason reported"
    )

    return (
        job_state,
        hold_reason
    )


def _query_condor_history(
    job_id: str,
    timeout: int = 30,
    interval: int = 1,
):
    """
    Query the final state of a job
    after it has left condor_q.

    Returns a dictionary containing:

        JobStatus
        ExitCode
        ExitBySignal
        ExitSignal
        RemoveReason
    """

    start = time.time()

    while True:

        result = subprocess.run(
            [
                "condor_history",
                job_id,
                "-limit",
                "1",
                "-af",
                "JobStatus",
                "ExitCode",
                "ExitBySignal",
                "ExitSignal",
                "RemoveReason",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "condor_history failed "
                f"for {job_id}\n"
                f"stdout:\n"
                f"{result.stdout}\n"
                f"stderr:\n"
                f"{result.stderr}"
            )

        line = (
            result.stdout.strip()
        )

        if line:

            parts = line.split(
                maxsplit=4
            )

            while len(parts) < 5:

                parts.append(
                    "undefined"
                )

            return {
                "JobStatus":
                    parts[0],

                "ExitCode":
                    parts[1],

                "ExitBySignal":
                    parts[2],

                "ExitSignal":
                    parts[3],

                "RemoveReason":
                    parts[4],
            }

        if (
            time.time()
            - start
            > timeout
        ):

            raise RuntimeError(
                f"Job {job_id} is no "
                "longer in condor_q, "
                "but no matching "
                "record appeared in "
                "condor_history within "
                f"{timeout} s."
            )

        time.sleep(
            interval
        )


def _remove_condor_jobs(
    job_ids
):
    """
    Best-effort cleanup.

    Only jobs submitted by this
    workflow are removed.
    """

    for job_id in job_ids:

        if not job_id:
            continue

        result = subprocess.run(
            [
                "condor_rm",
                job_id,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:

            logging.info(
                "Removed HTCondor "
                f"job {job_id} "
                "during cleanup."
            )

        else:

            logging.warning(
                "Could not remove "
                "HTCondor job "
                f"{job_id} during "
                "cleanup: "
                f"{result.stderr.strip()}"
            )


def condor_submit_with_rate_limit(
    sub_files: list,
    rate_limit: int = 20,
):

    job_status = {
        "Waiting":
            list(sub_files),

        "Running":
            [],

        "Done":
            [],
    }

    total_jobs = len(
        job_status["Waiting"]
    )

    try:

        while (
            len(
                job_status["Done"]
            )
            < total_jobs
        ):

            #
            # Submit jobs until the
            # configured concurrency
            # limit is reached.
            #
            while (
                job_status["Waiting"]
                and len(
                    job_status["Running"]
                )
                < rate_limit
            ):

                sub_file = (
                    job_status[
                        "Waiting"
                    ].pop(0)
                )

                logging.info(
                    f"Submitting "
                    f"{sub_file}"
                )

                job_id = (
                    submit_condor_job(
                        sub_file=sub_file
                    )
                )

                job_status[
                    "Running"
                ].append(
                    (
                        sub_file,
                        job_id
                    )
                )

            #
            # Nothing is running.
            #
            if not job_status[
                "Running"
            ]:

                if job_status[
                    "Waiting"
                ]:
                    continue

                break

            #
            # Poll HTCondor.
            #
            time.sleep(10)

            still_running = []

            for (
                sub_file,
                job_id
            ) in job_status[
                "Running"
            ]:

                (
                    job_state,
                    hold_reason
                ) = _query_condor_job(
                    job_id
                )

                #
                # Temporary submit-side
                # security/connectivity
                # issue.
                #
                if (
                    job_state
                    == "SECMAN"
                ):

                    still_running.append(
                        (
                            sub_file,
                            job_id
                        )
                    )

                    continue

                #
                # HTCondor JobStatus:
                #
                # 1 = Idle
                # 2 = Running
                # 3 = Removing
                # 4 = Completed
                # 5 = Held
                # 6 = Transferring Output
                # 7 = Suspended
                #
                if job_state == 5:

                    raise RuntimeError(
                        "HTCondor job "
                        f"{job_id} is held.\n"
                        "HoldReason: "
                        f"{hold_reason}\n"
                        "Submit file: "
                        f"{sub_file}"
                    )

                #
                # Job is still present
                # in condor_q.
                #
                if job_state is not None:

                    still_running.append(
                        (
                            sub_file,
                            job_id
                        )
                    )

                    continue

                #
                # Job has disappeared
                # from condor_q.
                #
                # IMPORTANT:
                # Do not interpret this
                # as success.
                #
                # Verify condor_history.
                #
                history = (
                    _query_condor_history(
                        job_id
                    )
                )

                job_completed = (
                    history[
                        "JobStatus"
                    ]
                    == "4"

                    and history[
                        "ExitCode"
                    ]
                    == "0"

                    and history[
                        "ExitBySignal"
                    ].lower()
                    == "false"
                )

                if not job_completed:

                    raise RuntimeError(
                        "HTCondor job "
                        f"{job_id} did "
                        "not complete "
                        "successfully.\n"
                        "Final status: "
                        f"{history}\n"
                        "Submit file: "
                        f"{sub_file}"
                    )

                #
                # The current GWAK
                # worker can sometimes
                # return exit code 0
                # even if one of its
                # child inference
                # processes failed.
                #
                # Therefore job.err
                # remains an additional
                # safety check.
                #
                error_file = (
                    Path(
                        sub_file
                    ).parent
                    / "job.err"
                )

                wait_for_file(
                    error_file
                )

                if (
                    os.path.getsize(
                        error_file
                    )
                    != 0
                ):

                    raise RuntimeError(
                        "HTCondor job "
                        f"{job_id} exited "
                        "with code 0, but "
                        "stderr is not "
                        "empty:\n"
                        f"{error_file}\n"
                        "This may indicate "
                        "an inference / "
                        "Triton failure "
                        "that was not "
                        "propagated as a "
                        "non-zero exit "
                        "code."
                    )

                logging.info(
                    f"Job {job_id} "
                    "completed "
                    "successfully "
                    "(ExitCode=0, "
                    "no signal, "
                    "empty stderr)."
                )

                job_status[
                    "Done"
                ].append(
                    sub_file
                )

            #
            # Replace the running list
            # only after the iteration
            # has completed.
            #
            job_status[
                "Running"
            ] = still_running

    except Exception:

        #
        # Fail-fast.
        #
        # Remove only jobs belonging
        # to this workflow.
        #
        active_job_ids = [
            job_id
            for _, job_id
            in job_status[
                "Running"
            ]
        ]

        if active_job_ids:

            logging.error(
                "Inference workflow "
                "failed. Removing "
                "remaining active "
                "HTCondor jobs: "
                f"{active_job_ids}"
            )

            _remove_condor_jobs(
                active_job_ids
            )

        raise

    logging.info(
        "All "
        f"{len(job_status['Done'])}"
        f"/{total_jobs} "
        "HTCondor jobs completed "
        "successfully."
    )
