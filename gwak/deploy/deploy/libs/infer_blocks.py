import h5py
import time
import logging
import psutil
import socket

import numpy as np

from pathlib import Path

from hermes.aeriel.client import InferenceClient



def get_ip_address() -> str:
    """
    Get the local nodes cluster-internal IP address
    """
    for _, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if (
                addr.family == socket.AF_INET
                and not addr.address.startswith("127.")
            ):
                
                return addr.address


def read_h5_data(
    test_data_dir: Path,
    key: str = "data"
):
    """
    test_data_dir: A directory that contains a list of hdf file. 
    """

    data_files= sorted(test_data_dir.glob("*.h5"))

    data_list = []
    for file in data_files: 

        with h5py.File(file, "r") as h1:
            data = h1[key][:]

        data_list.append(data.astype("float32"))

    return data_list


def static_infer_process(
    batcher,
    num_ifo, 
    psd_length,
    kernel_length,
    fduration,
    stride_batch_size,
    sample_rate, 
    client: InferenceClient,
    patient: float=1e-1,  
    loop_verbose: int=100
): 
    """
    The client need to connect to Triton serve already
    """
    results = []

    segment_size = int((psd_length + kernel_length + fduration + stride_batch_size - 1) * sample_rate)

    for i, background in enumerate(batcher):

        if i % loop_verbose == 0: 
            logging.info(f"Producing inference result on {i}th iteration!")

        background = background.reshape(-1, num_ifo, segment_size)
        client.infer(background,request_id=i)

        # Wait for the Queue to return the result
        time.sleep(patient)
        result = client.get()
        while result is None:

            result = client.get()
        results.append(result[0])

    return results

