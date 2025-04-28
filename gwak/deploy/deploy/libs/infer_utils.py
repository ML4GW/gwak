import re
import h5py
import time
import yaml
import logging
import psutil
import socket

import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from hermes.aeriel.client import InferenceClient



EXTREME_CCSN = [
    "Pan_2021/FR",
    "Powell_2020/y20"
]

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


def load_h5_as_dict(
    chosen_signals: Path,
    source_file: Path
)-> dict:
    """Open up a buffer to load in different CCSN wavefroms.

    Args:
        chosen_signals (Path): A file with names of each wavefrom.
        source_file (Path): The path that contains reasmpled raw waveform.

    Returns:
        dict: Time and resampled SQDM of Each waveform
    """
    with open(chosen_signals) as f:
        selected_ccsn = yaml.load(f, Loader=yaml.SafeLoader)
        
    source_file = Path(source_file)
    
    grand_dict = {}
    ccsn_list = []
    
    for key in selected_ccsn.keys():
        
        for name in selected_ccsn[key]:
            ccsn_list.append(f"{key}/{name}")

    for name in ccsn_list:
        
        with h5py.File(source_file/ f'{name}.h5', 'r', locking=False) as h:

            time = np.array(h['time'][:])
            quad_moment = h['quad_moment'][:] 
            
        grand_dict[name] =  [time, quad_moment]
    
    return grand_dict


def get_hp_hc_from_q2ij( 
    q2ij, 
    theta: np.ndarray, 
    phi: np.ndarray
):

    '''
    The orientation of GW emition is given by theta, phi
    '''

    hp =\
        q2ij[:,0,0]*(np.cos(theta)**2*np.cos(phi)**2 - np.sin(phi)**2).reshape(-1, 1)\
        + q2ij[:,1,1]*(np.cos(theta)**2*np.sin(phi)**2 - np.cos(phi)**2).reshape(-1, 1)\
        + q2ij[:,2,2]*(np.sin(theta)**2).reshape(-1, 1)\
        + q2ij[:,0,1]*(np.cos(theta)**2*np.sin(2*phi) - np.sin(2*phi)).reshape(-1, 1)\
        - q2ij[:,1,2]*(np.sin(2*theta)*np.sin(phi)).reshape(-1, 1)\
        - q2ij[:,2,0]*(np.sin(2*theta)*np.cos(phi)).reshape(-1, 1)

    hc = 2*(
        - q2ij[:,0,0]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,1,1]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,0,1]*(np.cos(theta)*np.cos(2*phi)).reshape(-1, 1)
        - q2ij[:,1,2]*(np.sin(theta)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,2,0]*(np.sin(theta)*np.sin(phi)).reshape(-1, 1)
    )

    return hp, hc


def on_grid_pol_to_sim(quad_moment, sqrtnum=16):

    CosTheta = np.linspace(-1, 1, sqrtnum)
    theta = np.arccos(CosTheta)
    phi = np.linspace(0, 2*np.pi, sqrtnum)

    theta = theta.repeat(sqrtnum)
    phi = np.tile(phi, sqrtnum)

    hp, hc = get_hp_hc_from_q2ij(quad_moment, theta, phi)

    return hp, hc, theta, phi


# To do add masking window to the function
def padding(
    time,
    hp,
    hc,
    distance,
    sample_kernel = 3,
    sample_rate = 4096,
    time_shift = -0.15, # shift zero to distination time
):
    
    # Two polarization
    signal = np.zeros([hp.shape[0], 2, sample_kernel * sample_rate])

    half_kernel_idx = int(sample_kernel * sample_rate/2)
    time_shift_idx = int(time_shift * sample_rate)
    t0_idx = int(time[0] * sample_rate)

    start = half_kernel_idx + t0_idx + time_shift_idx
    end = half_kernel_idx + t0_idx + time.shape[0] + time_shift_idx

    signal[:, 0, start:end] = hp / distance.reshape(-1, 1)
    signal[:, 1, start:end] = hc / distance.reshape(-1, 1)
    
    return signal

def get_seg_start_end(path):
    path = Path(path)  
    stem = path.stem    
    parts = stem.replace("background-", "").split("-")
    seg_start = int(parts[0])
    seg_end = int(parts[1])
    return seg_start, seg_end