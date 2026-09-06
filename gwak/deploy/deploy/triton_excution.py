import os
import h5py
import logging
import time 
import numpy as np

from typing import Union
from pathlib import Path
from jsonargparse import ArgumentParser, ActionConfigFile

from hermes.aeriel.client import InferenceClient
from infer_data import Sequence, CCSN_Waveform_Projector, load_h5_as_dict
from deploy.libs import gwak_logger, get_seg_start_end


EXTREME_CCSN = [
    "Pan_2021/FR",
    "Powell_2020/y20"
]

def run_infer(
    job_dir: Path,
    result_dir: Path,
    triton_server_ip: str,
    gwak_streamer: str,
    sequence_id: int,
    strain_file: Union[str, Path],
    data_format: str,
    grpc_port: int = 9001,
    shifts: list=[0.0, 1.0],
    psd_length: float=64,
    stride_batch_size:int=256,
    ifos:list=["H1", "L1"],
    kernel_size:int=2048,
    sample_rate:int=4096,
    inference_sampling_rate:float=1,
    dim_split: list=[6,1,1],
    **kwargs
):

    # File and Path management
    gwak_logger(job_dir / "log.log")
    roll_idx = np.cumsum(dim_split)
    seg_start, seg_end = get_seg_start_end(strain_file)
    if shifts[1] >= seg_end:
        logging.warning("Shift is larger then the entire backgorund data.")
        logging.warning("Skip the process.")
        import sys
        sys.exit()
    file_name = f"sequence_{seg_start}-{seg_end}_{int(shifts[1])}.h5"
    result_file = result_dir / file_name
    
    logging.info(f"Applying shifts = {shifts} to {strain_file}")
    sequence = Sequence(
        fname=strain_file,
        data_format=data_format,
        shifts=shifts,
        psd_length=psd_length,
        stride_batch_size=stride_batch_size,
        ifos=ifos,
        kernel_size=kernel_size,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        inj_type=None,
        dim_sum=sum(dim_split),
    )
    logging.info(f"Strain data loaded.")
    # Triton setup
    client = InferenceClient(
        address=f"{triton_server_ip}:{grpc_port}", 
        model_name=gwak_streamer,
        callback=sequence,
    )

    with client:

        for i, (bh_state, _) in enumerate(sequence):
            
            sequence_start = (i == 0)
            sequence_end = (i == (len(sequence) - 1))

            # Sending request to the triton server. 
            client.infer(
                np.stack([bh_state, bh_state]),
                request_id=i,
                sequence_id=sequence_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
             )
             
            if not i:
                while not sequence.started:
                    client.get()
                    time.sleep(1e-2)
        # Retrieve response from the triton server. 
        result = client.get()
        while result is None:
            result = client.get()
        # Make a function for this to connect between combine model and here.
        # Are there __names__ for this type of setup regardding to the output format

        gwak_value = result[0]
        print(f"{gwak_value.shape = }")
        # embedding = result[0][:, 0:roll_idx[0]]
        # f_coh = result[0][:, roll_idx[0]:roll_idx[1]]
        # gwak_value = result[0][:, roll_idx[1]:roll_idx[2]]

    logging.info(f"Collecting result to {result_file.resolve()}")

    with h5py.File(result_file, "w") as f:
        # f.create_dataset(f"embedding", data=embedding)
        # f.create_dataset(f"f_coh", data=f_coh)
        f.create_dataset(f"gwak_value", data=gwak_value)

        try:
            f.attrs["GPS_start"] = sequence.gps_start
        except Exception as e:
            logging.info(f"{type(e).__name__}") 

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(run_infer)

    args = parser.parse_args()
    args = args.as_dict()
    
    run_infer(**args)
