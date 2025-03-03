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

def read_gwpy_frames(
    test_data_dir: Path,
):
    """
    Reads a list of GWPy frame files from a directory and extracts data
    for the given channel. Returns a list of numpy arrays.

    Parameters
    ----------
    test_data_dir : Path
        Directory that contains GWPy frame files (e.g. *.gwf).
    channel : str
        The channel name to read from each file. For example: 'H1:GWOSC-16KHZ_R1_STRAIN'.

    Returns
    -------
    data_list : list of np.ndarray
        List of arrays containing the time-series data for each file.
    """
    data_files = sorted(test_data_dir.glob("*.gwf"))
    data_list = []

    for file in data_files:
        # Read the time series from the specified channel
        ts_h1 = TimeSeries.read(file, channel="H1:STRAIN_BURST_0")
        ts_l1 = TimeSeries.read(file, channel="L1:STRAIN_BURST_0")
        # Convert the data values to float32 if desired
        data_list.append([ts_h1.value.astype("float32"), ts_l1.value.astype("float32")])

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

