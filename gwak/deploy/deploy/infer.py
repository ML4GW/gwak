import time
import h5py
import logging
import subprocess

import numpy as np

from tqdm import tqdm
from zlib import adler32
from pathlib import Path

from hermes.aeriel.serve import serve
from hermes.aeriel.client import InferenceClient

from deploy.libs.infer_utils import get_ip_address 
from deploy.libs import gwak_logger
from deploy.libs.condor_tools import make_infer_config, make_subfile, submit_condor_job, condor_submit_with_rate_limit
from infer_data import get_shifts_meta_data, Sequence


def infer(
    ifos: list[str],
#     psd_length: float,
#     fduration: float,
    kernel_length: float,
    Tb: int,
    batch_size: int,
    stride_batch_size: int,
    sample_rate: int,
    fname: Path, 
    ccsn_repo: Path,
    data_format: str,
    shifts: list[float], 
    project: str,
    model_repo_dir: Path,
    image: Path,
    result_dir: Path,
    rate_limit: int = 20,
    load_model_patients: int=10,
    inference_sampling_rate: float = 1,
    **kwargs,
):
    
    result_dir = result_dir / project
    print('RESULTSDIR', result_dir)
    model_repo_dir = model_repo_dir / project

    log_file = result_dir / "log.log"
    triton_log = result_dir / "triton.log"
    result_dir.mkdir(parents=True, exist_ok=True)

    gwak_logger(log_file)

    #ip = get_ip_address()
    ip = '10.14.0.121'
    kernel_size = int(kernel_length * sample_rate * stride_batch_size)
    gwak_streamer = f"gwak-{project}-streamer"

    # Data handler
    logging.info(f"Loading data files ...")
    logging.info(f"    Data directory at {fname}")

    num_shifts, fnames, segments = get_shifts_meta_data(fname, Tb, shifts)

    serve_context = serve(
        model_repo_dir, 
        image, 
        log_file=triton_log, 
        wait=False
    )   

    with serve_context:
        
        logging.info(f"Waiting {load_model_patients} seconds to load model to triton!")
        time.sleep(load_model_patients)

        submit_num = 0
        sub_files = []

        start = time.time()
        for fname, (seg_start, seg_end) in zip(fnames, segments):
            for shift in range(num_shifts):

                fingerprint = f"{seg_start}{seg_end}{shift}".encode()
                sequence_id = adler32(fingerprint)

                _shifts = [s * (shift + 1) for s in shifts]
                job_dir = result_dir / f"condor/job_{submit_num:03d}"


                config_file = make_infer_config(
                    job_dir=job_dir,
                    triton_server_ip=ip,
                    gwak_streamer=gwak_streamer,
                    sequence_id=sequence_id,
                    strain_file=fname, 
                    data_format=data_format,
                    shifts=_shifts,
                    batch_size=batch_size,
                    ifos=ifos,
                    kernel_size=kernel_size,
                    inference_sampling_rate=inference_sampling_rate,
                )

                submit_file = make_subfile(
                    job_dir=job_dir,
                    arguments=Path("deploy/triton_excution.py").resolve(),
                    config=config_file.resolve()
                )

                sub_files.append(submit_file)
                submit_num += 1

        condor_submit_with_rate_limit(
            sub_files=sub_files,
            rate_limit=rate_limit
        )

        logging.info(f"Time spent for inference: {(time.time() - start)/60:.02f}mins")