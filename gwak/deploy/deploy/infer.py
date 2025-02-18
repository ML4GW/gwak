import time
import h5py
import logging

import numpy as np

from pathlib import Path

from hermes.aeriel.serve import serve
from hermes.aeriel.client import InferenceClient

from libs.infer_blocks import get_ip_address, read_h5_data, static_infer_process
from deploy.libs import gwak_logger

def infer(
    num_ifos: int,
    psd_length: float,
    fduration: float,
    kernel_length: float,
    batch_size: int,
    stride_batch_size: int,
    sample_rate: int,
    test_data_dir: Path, 
    project:Path,
    model_repo_dir: Path,
    image: Path,
    result_dir: Path,
    load_model_patients: int=10,
    **kwargs,
):

    result_dir = result_dir / project
    model_repo_dir = model_repo_dir / project
    result_dir.mkdir(parents=True, exist_ok=True)

    log_file = result_dir / "log.log"
    triton_log = result_dir / "triton.log"
    result_file = result_dir / "result.h5"

    gwak_logger(log_file)

    ip = get_ip_address()
    address=f"{ip}:8001"
    
    whiten_model = "preprocessor"
    gwak_model = f"gwak-{project}"

    serve_context = serve(
        model_repo_dir, 
        image, 
        log_file=triton_log, 
        wait=False
    )

    logging.info(f"Loading data files ...")
    logging.info(f"    Data directory at {test_data_dir}")
    data_list = read_h5_data(
        test_data_dir = test_data_dir, 
        key="data"
    )
    # data_list = [np.random.normal(0, 1, (256, 2, 133320)).astype("float32") for _ in range(2)]

    with serve_context:

        # Wait for the serve to connect!
        logging.info(f"Waiting {load_model_patients} seconds to load model to triton!")
        time.sleep(load_model_patients)

        client_1 = InferenceClient(address, whiten_model)
        client_2 = InferenceClient(address, gwak_model)

        # Produce whiten strain
        whitened_data = static_infer_process(
            batcher=data_list,
            num_ifo=num_ifos, 
            psd_length=psd_length,
            fduration=fduration,
            kernel_length=kernel_length,
            stride_batch_size=stride_batch_size,
            sample_rate=sample_rate, 
            client=client_1,
        )

        # Produce inference result
        inference_result = static_infer_process(
            batcher=whitened_data,
            num_ifo=num_ifos, 
            psd_length=0,
            fduration=0,
            kernel_length=kernel_length,
            stride_batch_size=stride_batch_size,
            sample_rate=sample_rate, 
            client=client_2
        )
    inference_results = np.stack(inference_result)

    logging.info(f"Collecting result to {result_file.resolve()}")
    with h5py.File(result_file, "w") as h:
        h.create_dataset(f"{project}", data=inference_results)

