import re
import os, sys
import logging
import subprocess


from typing import Union
from pathlib import Path



def make_subfile(
    filename,
    arguments,
    config
):
    
    condor_config = {}
    
    condor_config["universe"] = "vanilla"
    condor_config["executable"] = sys.executable
    condor_config["arguments"] = f"{arguments} --config {config}"
    
    
    condor_config["log"] = "job.log"
    condor_config["output"] = "job.out"
    condor_config["error"] = "job.err"
    
    # condor_config["getenv"] = True
    condor_config["environment"] = f"PYTHONPATH={os.environ.get('PYTHONPATH')}; \PATH={os.environ.get('PATH')}"
    
    
    condor_config["request_cpus"] = 1
    condor_config["request_memory"] = "512M"
    condor_config["request_disk"] = "1G"
    condor_config["accounting_group"] = "ligo.dev.o4.cbc.explore.test"
    
    with open(filename, "w") as f:
        for key, value in condor_config.items():
            f.write(f"{key} = {value}\n")
        
        f.write("queue 0")


def make_infer_config(
    config_file: Path,
    triton_server_ip,
    gwak_streamer, # gwak-white_noise_burst-streamer
    sequence_id,
    strain_file: Union[str, Path], # = "/home/hongyin.chen/Data/GWAK_data/gwak_infer_data/background-1240853610-680.hdf5"
    shifts:list,
    batch_size:int,
    ifos:list,
    kernel_size:int,
    sample_rate=2048,
    inference_sampling_rate=1,
    # inj_type=None,
): 

    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        for key, value in locals().items():  # Loop through all function arguments
            if key in ("config_file", "f"):
                continue

            f.write(f"{key}: {value}\n")  # Write each key-value pair
    




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
        logging.info(f"Job submitted successfully! Job ID: {job_id}")

        return job_id
    else:
        logging.info("Job submission failed or Job ID not found.")
        return None


def track_job(job_id):
    try:
        result = subprocess.run(["condor_q", job_id], capture_output=True, text=True)
        if result.stdout.strip():
            print(f"Job {job_id} is still running or in queue.")
            print(result.stdout)
        else:
            print(f"Job {job_id} is not in the queue (may have completed).")
    except Exception as e:
        print(f"Error checking job status: {e}")


def wait_for_job_completion(job_id, log_file="job.log"):
    try:
        logging.info(f"Waiting for job {job_id} to complete...")
        subprocess.run(["condor_wait", log_file, job_id], check=True)
        logging.info(f"Job {job_id} has completed!")
    except subprocess.CalledProcessError:
        logging.info(f"Error while waiting for job {job_id}.")
        
        

def wait_for_jobs_popen(jobs):
    """
    Wait for multiple Condor jobs in parallel using subprocess.Popen.
    
    :param jobs: List of tuples (log_file, job_id).
    """
    processes = []
    
    for log_file, job_id in jobs:
        logging.info(f"Starting condor_wait for job {job_id} (log: {log_file})...")
        p = subprocess.Popen(["condor_wait", str(log_file), job_id])
        processes.append((p, log_file, job_id))
    
    # Wait for all processes to complete
    for proc, log_file, job_id in processes:
        proc.wait()  # Blocks until the process finishes
        if proc.returncode == 0:
            logging.info(f"Job {job_id} (log: {log_file}) completed.")
        else:
            logging.error(f"Job {job_id} (log: {log_file}) failed with exit code {p.returncode}.")
