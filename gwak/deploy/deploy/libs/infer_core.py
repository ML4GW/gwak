import time
import logging
import subprocess
from zlib import adler32
from pathlib import Path
from hermes.aeriel.serve import serve
from hermes.aeriel.monitor import ServerMonitor
from deploy.libs.infer_utils import get_ip_address
from deploy.libs.cluster_tools import make_infer_config

def bash_commnad_files(bash_file, command):
    bash_file = bash_file / "triton.sh"
    with open(bash_file, "w") as f:
        f.write(command)

    return bash_file


def run_bash(bash_file):

    subprocess.run(
        ["bash", f"{bash_file}"],  
    )


def client_action(
    fnames,
    segments,
    num_shifts,
    shifts:list,
    Tb: int,
    job_dir,
    result_dir,
    ip,
    grpc_port,
    gwak_streamer,
    data_format,
    psd_length,
    stride_batch_size,
    ifos,
    kernel_size,
    sample_rate,
    inference_sampling_rate,
    arguments,
    job_tag=None
):
    

    submit_count = 0
    bash_files = []

    # Make config
    for fname, (seg_start, seg_end) in zip(fnames, segments):
        for shift in range(num_shifts):

            fingerprint = f"{seg_start}{seg_end}{shift}".encode()
            if job_tag is not None:
                fingerprint = f"{seg_start}{seg_end}{shift}{job_tag}".encode()
            sequence_id = adler32(fingerprint)
            _shifts = [s * (shift + 1) for s in shifts]
            if Tb == 0: 
                _shifts = [0, 0]
            resolved_job_dir = job_dir / f"batch_job/job_{submit_count:07d}" # Make this to flexable
            _result_dir = Path(result_dir) / "inference_result" # infer_result_dir
            job_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating config at {resolved_job_dir}.")
            config_file = make_infer_config(
                job_dir=resolved_job_dir, #local
                result_dir=_result_dir,#local 
                triton_server_ip=ip,
                grpc_port=grpc_port,
                gwak_streamer=gwak_streamer,
                sequence_id=sequence_id, #local
                strain_file=fname, 
                data_format=data_format,
                shifts=_shifts, # local
                psd_length=psd_length,
                stride_batch_size=stride_batch_size,
                ifos=ifos,
                kernel_size=kernel_size,
                sample_rate=sample_rate,
                inference_sampling_rate=inference_sampling_rate,
            )

            cmd = f"python {str(arguments)} --config {config_file}"
            bash_files.append(bash_commnad_files(resolved_job_dir, cmd))

            submit_count += 1
            
    return bash_files