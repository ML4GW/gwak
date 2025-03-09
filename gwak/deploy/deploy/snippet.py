import h5py
import logging

import numpy as np

from typing import Union
from pathlib import Path
from jsonargparse import ArgumentParser, ActionConfigFile

from hermes.aeriel.client import InferenceClient
from infer_data import get_shifts_meta_data, Sequence
from deploy.libs import gwak_logger

gwak_logger()

def run_infer(
    triton_server_ip: str,
    gwak_streamer: str,
    sequence_id: int,
    strain_file: Union[str, Path],
    shifts=[0.0, 1.0],
    batch_size:int=1,
    ifos:list=["H1", "L1"],
    kernel_size:int=2048,
    sample_rate:int=2048,
    inference_sampling_rate:float=1,
    # inj_type=None,
    **kwargs
):


    client = InferenceClient(f"{triton_server_ip}:8001", gwak_streamer)

    results = []
    with client:

        sequence = Sequence(
            fname=strain_file,
            shifts=shifts,
            batch_size=batch_size,
            ifos=ifos,
            kernel_size=kernel_size,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            inj_type=None,
        )
        
        for i, (bh_state, inj_state) in enumerate(sequence):

            sequence_start = (i == 0)
            sequence_end = (i == (len(sequence) - 1))

            # Sending request to the triton server. 
            client.infer(
                bh_state,
                request_id=i,
                sequence_id=sequence_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )
            
            if inj_state is not None:
                
                print(inj_state)

            # Retrieve response from the triton server. 
            result = client.get()
            while result is None:
                result = client.get()

            results.append(result[0])
        # Job Done leaving client 

    results = np.stack(results)
    result_file = Path(f"sequence_{sequence_id}.h5")
    logging.info(f"Collecting result to {result_file.resolve()}")

    with h5py.File(result_file, "w") as f:
        f.create_dataset(f"data", data=results)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(run_infer)

    args = parser.parse_args()
    args = args.as_dict()
    
    run_infer(**args)
    