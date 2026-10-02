
import re
import io
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
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(interval)

def write_bash_file(
    bash_root: Path,
    files: list,
    command: str
):

    bash_file = bash_root / "condor_cmd.sh"
    gwak_root = Path(os.getenv('GWAK_ROOT'))
    gwak_env = gwak_root / ".gwak/env.sh"
    deploy_env = gwak_root / "gwak/deploy/.venv/bin/activate"

    with bash_file.open("w") as sh_file:

        sh_file.write("#!/bin/bash\n")
        sh_file.write("set -euo pipefail\n\n")
        sh_file.write("export HDF5_USE_FILE_LOCKING=FALSE\n")
        for file in files:
            sh_file.write(f'cp "{file}" "$_CONDOR_SCRATCH_DIR/"\n')

        sh_file.write(f'source "{gwak_env}"\n')
        sh_file.write(f'source "{deploy_env}"\n')
        sh_file.write(f"exec {command}\n")
    bash_file.chmod(0o755)
    return bash_file


def write_export_config(
    **export_kwargs
):

    # Load the arguments in main eport file
    with open(export_kwargs["export_config_file"], "r") as yaml_file:
        export_args = yaml.safe_load(yaml_file)

    # Replace the arguments for inference
    for key, item in export_kwargs.items():

        if key in ("export_config_file", "export_job_dir"):
            continue
        export_args[key] = item

    # Write the and save the new arguments
    export_config = Path(export_kwargs["export_job_dir"]) / "export.yaml"
    with open(export_config, "w") as yaml_file:

        for key, value in export_args.items():
            if value is None:
                value = "null"
            yaml_file.write(f"{key}: {value}\n")

    return export_config


def write_condor_config(
    condor_kwargs,
    job_dir,
    executable,
    config
):

    condor_config = {}
    submit_file = job_dir / "condor.sub"
    job_out = job_dir / "job.out"
    job_out.touch()

    condor_config["universe"] = "vanilla"
    condor_config["executable"] = executable

    condor_config["log"] = job_dir / "job.log"
    condor_config["output"] = job_out
    condor_config["error"] = job_dir / "job.err"

    for key in condor_kwargs.keys():
        condor_config[key] = condor_kwargs[key]

    with open(submit_file, "w") as f:
        for key, value in condor_config.items():
            f.write(f"{key} = {value}\n")

        f.write("queue")

    return submit_file


def write_slurm_config(
    kwargs,
    job_dir,
    export_config,
    infer_config,
):

    file_path = Path(__file__).resolve()
    filename = job_dir / "submit.slurm" 
    deploy_app_path = file_path.parents[2]

    # Deploy commands
    export_cmd = (
        f"{kwargs['deploy_cmd']['export'][0]} "
        f"--config {export_config} " # Modify this to follow run dir
        # f"--project {project} "
        # f"--output_dir {job_dir}/export"
    )

    infer_cmd = (
        f"{kwargs['deploy_cmd']['infer'][0]} "
        f"--config {infer_config}"
    )

    # GPU setting
    kwargs["gres"] = f"gpu:{kwargs['gpu_per_node']}"
    if kwargs["gpu_card"] is not None:
        kwargs["gres"] = f"gpu:{kwargs['gpu_card']}:{kwargs['gpu_per_node']}"


    config_content = ["#!/bin/bash"]
    for item, key in kwargs.items():

        if item in (
            "gpu_card", 
            "gpu_per_node", 
            "cmd", 
            "deploy_cmd"
        ):
            continue

        config_content.append(f"#SBATCH --{item}={key}")

    config_content.append("")
    config_content.append(f"cd {deploy_app_path}")
    cmds = kwargs.get("cmd") or []
    for cmd in cmds:
        config_content.append(cmd)
    config_content.append(export_cmd)
    config_content.append(infer_cmd)
    config_content.append("")

    with open(filename, "w") as f:
        f.write("\n".join(config_content))

    print(f"SLURM script written to: {filename}")
    return filename

def write_infer_core_config(
    **infer_core_kwargs
):

    yaml_file = Path(infer_core_kwargs["job_dir"]) / "config.yaml"

    with open(yaml_file, "w") as f:
        for key, value in infer_core_kwargs.items():
            if key in ("fnames") and isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {Path(item).name}\n")

            elif key in ("segments") and isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")

            else:
                if value is None:
                    value = "null"
                f.write(f"{key}: {value}\n")

    return yaml_file

def write_infer_config(
    job_dir: Path,
    result_dir: Path,
    triton_server_ip,
    grpc_port,
    gwak_streamer,
    sequence_id,
    strain_file: Union[str, Path],
    data_format: str,
    shifts:list,
    psd_length:float,
    stride_batch_size:int,
    ifos:list,
    kernel_size:int,
    dim_split:list,
    sample_rate=2048,
    inference_sampling_rate=1,
):

    job_dir.mkdir(parents=True, exist_ok=True)
    config_file = job_dir / "config.yaml"

    with open(config_file, "w") as f:
        for key, value in locals().items():  # Loop through all function arguments
            if key in ("config_file", "f"):
                continue

            f.write(f"{key}: {value}\n")  # Write each key-value pair

    return config_file

def submit_condor_job(sub_file:Path):

    result = subprocess.run(
        ["condor_submit", str(sub_file)],
        cwd=sub_file.parent,
        capture_output=True, 
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"condor_submit failed for {sub_file}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    match = re.search(r"submitted to cluster (\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            "condor_submit returned success but no cluster ID was found for "
            f"{sub_file}\nstdout:\n{result.stdout}"
        )

    job_id = match.group(1) + ".0"
    logging.info(f"Job {job_id} submitted successfully!")
    return job_id


def _query_condor_job(job_id: str):
    """Return the current HTCondor status and hold reason."""

    result = subprocess.run(
        ["condor_q", job_id, "-af", "JobStatus", "HoldReason"],
        capture_output=True,
        text=True
    )

    if "SECMAN" in result.stderr:
        logging.warning(
            f"Temporary HTCondor SECMAN error while checking {job_id}: "
            f"{result.stderr.strip()}"
        )
        return "SECMAN", None

    if result.returncode != 0:
        raise RuntimeError(
            f"condor_q failed for {job_id}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    job_info = result.stdout.strip()
    if not job_info:
        return None, None

    parts = job_info.split(maxsplit=1)
    try:
        job_state = int(parts[0])
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse JobStatus for {job_id}: {job_info}"
        ) from exc

    hold_reason = parts[1] if len(parts) > 1 else "No HoldReason reported"
    return job_state, hold_reason


def _query_condor_history(
    job_id: str,
    timeout: int = 30,
    interval: int = 1,
):
    """Return the final HTCondor status after a job leaves condor_q."""

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
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"condor_history failed for {job_id}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        line = result.stdout.strip()
        if line:
            parts = line.split(maxsplit=4)
            parts.extend(["undefined"] * (5 - len(parts)))
            return {
                "JobStatus": parts[0],
                "ExitCode": parts[1],
                "ExitBySignal": parts[2],
                "ExitSignal": parts[3],
                "RemoveReason": parts[4],
            }

        if time.time() - start > timeout:
            raise RuntimeError(
                f"Job {job_id} left condor_q but no matching "
                f"condor_history record appeared within {timeout} s."
            )

        time.sleep(interval)


def _remove_condor_jobs(job_ids):
    """Remove jobs submitted by this inference run."""

    for job_id in job_ids:
        if not job_id:
            continue

        result = subprocess.run(
            ["condor_rm", job_id],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logging.info(f"Removed HTCondor job {job_id} during cleanup.")
        else:
            logging.warning(
                f"Could not remove HTCondor job {job_id}: "
                f"{result.stderr.strip()}"
            )


def condor_submit_with_rate_limit(
    sub_files: list,
    rate_limit: int= 20
):
    """Submit Condor jobs up to the rate limit and verify final status."""

    job_status = {
        "Waiting": list(sub_files),
        "Running": [],
        "Done": []
    }

    total_jobs = len(job_status["Waiting"])

    try:
        while len(job_status["Done"]) < total_jobs:

            while job_status["Waiting"] and len(job_status["Running"]) < rate_limit:
                sub_file = job_status["Waiting"].pop(0)
                logging.info(f"Submitting {sub_file}")
                job_id = submit_condor_job(sub_file=sub_file)
                job_status["Running"].append((sub_file, job_id))

            if not job_status["Running"]:
                break

            time.sleep(10)
            still_running = []

            for sub_file, job_id in job_status["Running"]:
                job_state, hold_reason = _query_condor_job(job_id)

                if job_state == "SECMAN":
                    still_running.append((sub_file, job_id))
                    continue

                # HTCondor JobStatus 5 is Held.
                if job_state == 5:
                    raise RuntimeError(
                        f"HTCondor job {job_id} is held.\n"
                        f"HoldReason: {hold_reason}\n"
                        f"Submit file: {sub_file}"
                    )

                if job_state is not None:
                    still_running.append((sub_file, job_id))
                    continue

                history = _query_condor_history(job_id)
                job_completed = (
                    history["JobStatus"] == "4"
                    and history["ExitCode"] == "0"
                    and history["ExitBySignal"].lower() == "false"
                )

                if not job_completed:
                    raise RuntimeError(
                        f"HTCondor job {job_id} did not complete successfully.\n"
                        f"Final status: {history}\n"
                        f"Submit file: {sub_file}"
                    )

                error_file = Path(sub_file).parent / "job.err"
                wait_for_file(error_file)

                if os.path.getsize(error_file) != 0:
                    logging.warning(f"Error file not empty: {error_file}")

                logging.info(f"Job {job_id} completed successfully.")
                job_status["Done"].append(sub_file)

            job_status["Running"] = still_running

    except Exception:
        active_job_ids = [job_id for _, job_id in job_status["Running"]]

        if active_job_ids:
            logging.error(
                "Inference workflow failed. Removing remaining active "
                f"HTCondor jobs: {active_job_ids}"
            )
            _remove_condor_jobs(active_job_ids)

        raise

    logging.info(
        f"All {len(job_status['Done'])}/{total_jobs} "
        "HTCondor jobs completed successfully."
    )
