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
from deploy.libs import gwak_logger

gwak_logger("log.log")

EXTREME_CCSN = [
    "Pan_2021/FR",
    "Powell_2020/y20"
]

def run_infer(
    triton_server_ip: str,
    gwak_streamer: str,
    sequence_id: int,
    strain_file: Union[str, Path],
    data_format: str,
    shifts: list=[0.0, 1.0],
    batch_size:int=1,
    stride_batch_size:int=256,
    ifos:list=["H1", "L1"],
    kernel_size:int=2048,
    sample_rate:int=2048,
    inference_sampling_rate:float=1,
    **kwargs
):
    
    sequence_kernel_size = kernel_size # Temporal setup to get over config setting
    dis_grid_count = 4
    one_side_pad_length = 0.5
    kernel_length = 0.5
    kernel_size = int(kernel_length * sample_rate)
    os.getenv # Use this to get ccsn config file. 
    inj_type=None # Set injection type manully
    client = InferenceClient(f"{triton_server_ip}:8001", gwak_streamer)
    print(os.getenv("CCSN_FILE"))
    
    with client:

        sequence = Sequence(
            fname=strain_file,
            data_format=data_format,
            shifts=shifts,
            batch_size=batch_size,
            ifos=ifos,
            kernel_size=sequence_kernel_size,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            inj_type=None,
        )

        signals_dict = load_h5_as_dict(
            chosen_signals="/home/hongyin.chen/anti_gravity/gwak/gwak/deploy/deploy/config/ccsn.yaml",
            source_file="/home/hongyin.chen/Data/3DCCSN_PREMIERE/Resampled_2048"
        )

        # No additional time shift
        projector = CCSN_Waveform_Projector(
            ifos=ifos,
            sample_rate=sample_rate,
            sample_duration=3, 
        )

        for key_idx, key in enumerate(signals_dict.keys()):
            logging.info(f"Running {key} analysis")

            # Initaizing distance setup
            min_dis, max_dis = 0.5, 20
            if key in EXTREME_CCSN: 
                min_dis, max_dis = 5, 100
            sampled_distance = np.geomspace(min_dis, max_dis, dis_grid_count)

            for dis_count, dis in enumerate(sampled_distance):
                
                results = [] # Inference result of one sequence
                for i, (bh_state, inj_state) in enumerate(sequence):

                    sequence_start = (i == 0)
                    sequence_end = (i == (len(sequence) - 1))

                    # Sample orientation and sky location
                    padded_ccsne = projector(
                        time = signals_dict[key][0],
                        quad_moment = signals_dict[key][1],
                        ori_theta=np.zeros(stride_batch_size),
                        ori_phi=np.zeros(stride_batch_size),
                        dec=np.zeros(stride_batch_size),
                        psi=np.zeros(stride_batch_size),
                        phi=np.zeros(stride_batch_size),
                    )

                    # Sample start from the middle of the padded CCSN since no additainl time shift applied
                    ccsne = padded_ccsne[:,:,3072:3072 + int(kernel_length * sample_rate)] / dis

                    # Can be replace by reshape ccsne and add the two array.
                    for inj_idx, ccsn in enumerate(ccsne):

                        inj_start = int((one_side_pad_length + inj_idx) * sample_rate)
                        inj_end = inj_start + kernel_size
                        inj_state[:, inj_start: inj_end] += ccsn  

                    
                    client.infer(
                        np.stack([bh_state, inj_state]),
                        request_id=i,
                        sequence_id=sequence_id + 1 + dis_grid_count * key_idx + dis_count,
                        sequence_start=sequence_start,
                        sequence_end=sequence_end,
                    )

                    result = client.get()
                    while result is None:
                        result = client.get()
                    results.append(result[0])
                    
                    # One sequence ends leave the sequence loop
                # For each Distance, of each CCSN template we save one file
                
                results = np.stack(results)
                key_name = f"{key.replace('/', '_')}/DIS_{dis:05.1f}"
                saving_dir = Path(f"../../inference_INJ_result/{key.replace('/', '_')}")
                saving_dir.mkdir(parents=True, exist_ok=True)
                result_file = saving_dir / f"Inj_sequence_{sequence_id}.h5"
                logging.info(f"Collecting result to {result_file.resolve()}")

                with h5py.File(result_file, "a") as f:
                    f.create_dataset(key_name, data=results)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(run_infer)

    args = parser.parse_args()
    args = args.as_dict()
    
    run_infer(**args)
    