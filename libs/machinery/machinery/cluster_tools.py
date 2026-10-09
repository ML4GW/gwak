
import re
import io
import os
import time
import yaml
import logging
import subprocess

from typing import Union
from pathlib import Path
from textwrap import dedent
from machinery import gwak_logger, gwak_dir

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

    condor_config["universe"] = "vanilla"
    condor_config["executable"] = executable

    condor_config["log"] = job_dir / "job.log"
    condor_config["output"] = job_dir / "job.out"
    condor_config["error"] = job_dir / "job.err"

    for key in condor_kwargs.keys():
        condor_config[key] = condor_kwargs[key]

    with open(submit_file, "w") as f:
        for key, value in condor_config.items():
            f.write(f"{key} = {value}\n")

        f.write("queue")

    return submit_file


def write_container_condor_sub(
    job_dir,
    image,
    condor_kwargs,
    initialdir,
    executable,
    transfer_output_files="Outputs",
):
    """ Container universe submit file. The image is pulled from
    osdf:///igwn/cit/staging/$USER/Container/GWAK/{image}, files listed in
    transfer_output_files land back in job_dir (initialdir). Any key in
    condor_kwargs overrides the defaults below. """

    condor_config = {}
    submit_file = job_dir / "condor.sub"
    username = os.environ.get("USER")
    condor_config["universe"] = "container"
    condor_config["container_image"] = image

    condor_config["executable"] = executable
    condor_config["initialdir"] = initialdir
    # On the CIT shared filesystem the default (IF_NEEDED) skips transfer,
    # so the osdf:// image would never be fetched.
    condor_config["should_transfer_files"] = "YES"
    condor_config["when_to_transfer_output"] = "ON_EXIT"
    condor_config["transfer_output_files"] = transfer_output_files
    # condor_config["transfer_input_files"] = f"/home/{username}/.netrc"

    condor_config["use_oauth_services"] = "scitokens"
    condor_config["requirements"] = "HAS_SINGULARITY && SINGULARITY_CAN_USE_SIF"

    condor_config["log"] = job_dir / "job.log"
    condor_config["output"] = job_dir / "job.out"
    condor_config["error"] = job_dir / "job.err"

    condor_config["request_gpus"] = 1
    condor_config["gpus_minimum_capability"] = 8.0
    # gpus_maximum_capability = 13
    condor_config["gpus_minimum_memory"] = "16GB"

    for key in condor_kwargs.keys():
        condor_config[key] = condor_kwargs[key]

    with open(submit_file, "w") as f:
        for key, value in condor_config.items():
            f.write(f"{key} = {value}\n")

        f.write("queue")

    return submit_file


def _symlinks_killer() -> str:
    """Return a Bash EXIT trap that replaces symlinks with actual files."""
    return dedent("""\
        flatten_symlinks() {
            while IFS= read -r -d '' symlink; do
                if target=$(readlink -f "$symlink") && [[ -f "$target" ]]; then
                    cp --remove-destination "$target" "$symlink"
                else
                    rm -f "$symlink"
                fi
            done < <(find "$CONTAINER_OUTPUT_DIR" -type l -print0)
        }

        trap flatten_symlinks EXIT
    """)

def write_trainer_bash_file(
    job_dir: Path,
    ifo_mode: str,
    data_tag: str,
    prefix: str,
    osdf_data_root: str,
    wandb_mode: str = "offline",
    num_cores: int = 4,
):
    """ Bash executable for one condor_train job.

    Arguments:
        osdf_data_url -- OSDF collection holding {ifo_mode}/{data_tag},
            pulled into the EP scratch before training.
        wandb_mode -- WANDB_MODE inside the job; the EP has no wandb key.
    """

    bash_file = job_dir / "condor_train.sh"
    # Add the symlinks killer to the bash script
    header = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "SCRATCH=${_CONDOR_SCRATCH_DIR:-$PWD}",
        "GWAK_ROOT=${GWAK_ROOT:-/opt/gwak}",
        "",
        "# Update environment variables for GWAK directories",
        "export GWAK_OUTPUT_DIR=$SCRATCH/muted_dir",
        "export GWAK_DATA_DIR=$SCRATCH/Data",
        "export CONTAINER_OUTPUT_DIR=$SCRATCH/Outputs",
        "export GWAK_LOG_DIR=$CONTAINER_OUTPUT_DIR/logs",
        "export GWAK_LOUVRE_DIR=$CONTAINER_OUTPUT_DIR/louvre",
        "export IMAGE_DIR=$SCRATCH/images",
        f"export WANDB_MODE={wandb_mode}",
        "",
        f"OSDF_SRC={osdf_data_root}",
        f"DEST=$GWAK_DATA_DIR/{ifo_mode}/{data_tag}",
        "",
        "# Enforce snakemake to use the correct paths settings.",
        "cat > \"$SCRATCH/paths.yaml\" <<EOF",
        "paths:",
        "    gwak_root: $GWAK_ROOT",
        "    gwak_output_dir: $GWAK_OUTPUT_DIR",
        "    gwak_data_dir: $GWAK_DATA_DIR",
        "    gwak_log_dir: $GWAK_LOG_DIR",
        "    gwak_louvre_dir: $GWAK_LOUVRE_DIR",
        "    image_dir: $IMAGE_DIR",
        "    container_output_dir: $CONTAINER_OUTPUT_DIR",
        "EOF",
        "",
        f"mkdir -p \"$GWAK_DATA_DIR/{ifo_mode}\"",
        "mkdir -p \"$CONTAINER_OUTPUT_DIR\"",
    ]

    with bash_file.open("w") as sh_file:
        sh_file.write("\n".join(header))
        sh_file.write("\n")
        sh_file.write("\n")
        # Register cleanup handler
        sh_file.write("# Replace symlinks with actual files on exit\n")
        sh_file.write(_symlinks_killer())
        sh_file.write("\n")
        # Transfer data
        sh_file.write("# Transfer data from OSDF to local scratch\n")
        sh_file.write('pelican object get -r "$OSDF_SRC" "$DEST"\n')
        sh_file.write('du -sh "$DEST"\n')
        sh_file.write("\n")
        # Run training
        sh_file.write("# Run snakemake for the condor_train job\n")
        sh_file.write('cd "$GWAK_ROOT"\n')
        sh_file.write('snakemake --directory "$CONTAINER_OUTPUT_DIR" \\\n')
        sh_file.write('    --configfile "$SCRATCH/paths.yaml" \\\n')
        sh_file.write(f'    -c{num_cores} \\\n')
        sh_file.write(f'    "$CONTAINER_OUTPUT_DIR/models/{prefix}/combination/model_JIT.pt"\n')

    bash_file.chmod(0o755)
    return bash_file

def write_slurm_config(
    kwargs,
    job_dir,
    export_config,
    infer_config,
):

    # deploy_dir = gwak_dir(suffix="gwak/deploy")()
    # pyproject_toml = deploy_dir / "pyproject.toml"
    env_setting = gwak_dir()(append_path=".gwak/env.sh")
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
    config_content.append(f"source {env_setting}")
    cmds = kwargs.get("cmd") or []
    for cmd in cmds:
        config_content.append(cmd)
    config_content.append(export_cmd)
    config_content.append(infer_cmd)
    config_content.append("")

    with open(filename, "w") as f:
        f.write("\n".join(config_content))

    logging.info(f"SLURM script written to: {filename}")
    return filename

def write_infer_core_config(
    keep_fname_dir: bool = False,
    **infer_core_kwargs
):
    """ keep_fname_dir -- Write full strain paths instead of basenames. """

    yaml_file = Path(infer_core_kwargs["job_dir"]) / "config.yaml"

    with open(yaml_file, "w") as f:
        for key, value in infer_core_kwargs.items():
            if key == "fnames" and isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    item = Path(item) if keep_fname_dir else Path(item).name
                    f.write(f"  - {item}\n")

            elif key == "segments" and isinstance(value, list):
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
        ["condor_submit", sub_file], 
        cwd=sub_file.parent,
        capture_output=True, 
        text=True
    )

    # Extract job ID from output using regex
    match = re.search(r"submitted to cluster (\d+)", result.stdout)
    if match:
        
        job_id = match.group(1) + ".0"  # Format as "12345.0"
        logging.info(f"Job {job_id} submitted successfully!")

        return job_id
    else:
        logging.info("Job submission failed or Job ID not found.")
        return None


def condor_submit_with_rate_limit(
    sub_files: list,
    concurrent_node: int= 10
):

    job_status = {
        "Waiting": sub_files,
        "Running": [],
        "Done": []
    }
    
    total_jobs = len(job_status["Waiting"])

    while len(job_status["Done"]) < total_jobs :

        # Check if we need to submit new jobs
        if len(job_status["Running"]) < concurrent_node:
            try: 
                logging.info(f"Submitting {job_status['Waiting'][0]}")
                job_id = submit_condor_job(sub_file=job_status["Waiting"][0])

                # Add in to Running track list
                job_status["Running"].append((job_status["Waiting"][0], job_id))
                job_status["Waiting"].pop(0)
                continue
            except IndexError:
                pass

        check_held = subprocess.run(["condor_release", "-all"], capture_output=True, text=True)
        time.sleep(10)
        # Check if any job is done
        for idx, (sub_file, job_id) in enumerate(job_status["Running"]):
            result = subprocess.run(["condor_q", f"{job_id}"], capture_output=True, text=True)

            if "SECMAN" in result.stderr:
                time.sleep(1)
                continue
            if not (job_id in result.stdout):
                error_file = Path(sub_file).parent / "job.err"
                job_status["Done"].append(sub_file)
                job_status["Running"].pop(idx)
                
                wait_for_file(error_file)
                condor_success = (result.returncode == 0)
                triton_success = (os.path.getsize(error_file) == 0)

                if condor_success and triton_success:
                    logging.info(f"Job {job_id} ran successfully!")
                if not condor_success:
                    logging.error(f"Job {job_id} failed check: {sub_file}")
                if not triton_success:
                    logging.warning(f"Error file not empty: {error_file}")
            time.sleep(0.1)