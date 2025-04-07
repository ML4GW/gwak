
import re
import os, sys
import time
import logging
import subprocess


from typing import Union
from pathlib import Path



def make_subfile(
    job_dir,
    arguments,
    config
):
    
    condor_config = {}
    submit_file = job_dir / "condor.sub"

    condor_config["universe"] = "vanilla"
    condor_config["executable"] = sys.executable
    condor_config["arguments"] = f"{arguments} --config {config}"
    
    
    condor_config["log"] = "job.log"
    condor_config["output"] = "job.out"
    condor_config["error"] = "job.err"
    
    # condor_config["getenv"] = True
    condor_config["environment"] = f"PYTHONPATH={os.environ.get('PYTHONPATH')}; \PATH={os.environ.get('PATH')}"
    
    
    condor_config["request_cpus"] = 1
    condor_config["request_memory"] = "1G"
    condor_config["request_disk"] = "2G"
    condor_config["accounting_group"] = "ligo.dev.o4.cbc.explore.test"
    
    with open(submit_file, "w") as f:
        for key, value in condor_config.items():
            f.write(f"{key} = {value}\n")
        
        f.write("queue 0")

    return submit_file

def make_infer_config(
    job_dir: Path,
    triton_server_ip,
    gwak_streamer,
    sequence_id,
    strain_file: Union[str, Path],
    data_format: str,
    shifts:list,
    batch_size:int,
    ifos:list,
    kernel_size:int,
    sample_rate=2048,
    inference_sampling_rate=1,
    # inj_type=None,
): 

    job_dir.mkdir(parents=True, exist_ok=True)
    config_file = job_dir / "config.yaml"

    with open(config_file, "w") as f:
        for key, value in locals().items():  # Loop through all function arguments
            if key in ("job_dir", "config_file", "f"):
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
        logging.info(f"Job submitted successfully! Job ID: {job_id}")

        return job_id
    else:
        logging.info("Job submission failed or Job ID not found.")
        return None


def condor_submit_with_rate_limit(
    sub_files: list,
    rate_limit: int= 20
):

    job_status = {
        "Waiting": sub_files,
        "Running": [],
        "Done": []
    }
    
    total_jobs = len(job_status["Waiting"])

    while len(job_status["Done"]) < total_jobs :

        # Check if we need to submit new jobs
        if len(job_status["Running"]) < rate_limit:
            
            try: 
                logging.info(f"Submitting {job_status['Waiting'][0]}")
                job_id = submit_condor_job(sub_file=job_status["Waiting"][0])

                # Add in to Running track list
                job_status["Running"].append((job_status["Waiting"][0], job_id))
                job_status["Waiting"].pop(0)
                continue
            except IndexError:
                pass
                
        time.sleep(10)
        # Check if any job is done
        result = subprocess.run("condor_q", capture_output=True, text=True)
        for idx, (sub_file, job_id) in enumerate(job_status["Running"]):

            if not (job_id in result.stdout):

                job_status["Done"].append(sub_file)
                job_status["Running"].pop(idx)
                print(f"Job {job_id} done!")