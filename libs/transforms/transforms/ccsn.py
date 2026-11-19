import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from ml4gw import gw
from ml4gw.distributions import Cosine
from torch.distributions.uniform import Uniform

from .utils import hrrs_value


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
    phi: np.ndarray,
    xp=np,
):

    '''
    The orientation of GW emition is given by theta, phi
    xp is np, or torch for tensors on the GPU
    '''

    hp =\
        q2ij[:,0,0]*(xp.cos(theta)**2*xp.cos(phi)**2 - xp.sin(phi)**2).reshape(-1, 1)\
        + q2ij[:,1,1]*(xp.cos(theta)**2*xp.sin(phi)**2 - xp.cos(phi)**2).reshape(-1, 1)\
        + q2ij[:,2,2]*(xp.sin(theta)**2).reshape(-1, 1)\
        + q2ij[:,0,1]*(xp.cos(theta)**2*xp.sin(2*phi) - xp.sin(2*phi)).reshape(-1, 1)\
        - q2ij[:,1,2]*(xp.sin(2*theta)*xp.sin(phi)).reshape(-1, 1)\
        - q2ij[:,2,0]*(xp.sin(2*theta)*xp.cos(phi)).reshape(-1, 1)

    hc = 2*(
        - q2ij[:,0,0]*(xp.cos(theta)*xp.sin(phi)*xp.cos(phi)).reshape(-1, 1)
        + q2ij[:,1,1]*(xp.cos(theta)*xp.sin(phi)*xp.cos(phi)).reshape(-1, 1)
        + q2ij[:,0,1]*(xp.cos(theta)*xp.cos(2*phi)).reshape(-1, 1)
        - q2ij[:,1,2]*(xp.sin(theta)*xp.cos(phi)).reshape(-1, 1)
        + q2ij[:,2,0]*(xp.sin(theta)*xp.sin(phi)).reshape(-1, 1)
    )

    return hp, hc

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
    signal = np.zeros([hp.shape[0], 2, int(sample_kernel * sample_rate)])

    half_kernel_idx = int(sample_kernel * sample_rate/2)
    time_shift_idx = int(time_shift * sample_rate)
    t0_idx = int(time[0] * sample_rate)

    start = half_kernel_idx + t0_idx + time_shift_idx
    end = half_kernel_idx + t0_idx + time.shape[0] + time_shift_idx

    signal[:, 0, start:end] = hp / distance.reshape(-1, 1)
    signal[:, 1, start:end] = hc / distance.reshape(-1, 1)

    return signal


class CCSN_Injector:

    def __init__(
        self,
        ifos,
        signals_dict,
        sample_rate,
        sample_duration,
        buffer_duration = 4,
        off_set = 0.0,
        time_shift = -0.15
    ):

        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)

        self.signals = signals_dict
        self.ccsn_list = list(self.signals.keys())

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.buffer_duration = buffer_duration
        self.off_set = off_set
        self.time_shift = time_shift
        self.kernel_length = int(sample_duration * sample_rate)
        self.buffer_length = int(buffer_duration * sample_rate)
        self.max_center_offset = int((buffer_duration/2 - sample_duration - off_set) * sample_rate)

        if off_set <= -time_shift:

            logging.info(f"Core bounce siganl may leak out of sample kernel by {-time_shift - off_set}")

        self.bank = None

    def template_bank(self, device):
        # quadrupole moments of all waveforms, placed in the buffer
        # the same way as padding() does it. Shape (n_ccsn, 3, 3, buffer_length)
        if self.bank is None or self.bank.device != device:
            bank = np.zeros((len(self.ccsn_list), 3, 3, self.buffer_length))
            half_kernel_idx = int(self.buffer_duration * self.sample_rate / 2)
            time_shift_idx = int(self.time_shift * self.sample_rate)
            for i, name in enumerate(self.ccsn_list):
                time, quad_moment = self.signals[name]
                start = half_kernel_idx + int(time[0] * self.sample_rate) + time_shift_idx
                bank[i, :, :, start:start + time.shape[0]] = np.moveaxis(quad_moment, 0, -1)
            self.bank = torch.as_tensor(bank, dtype=torch.float32, device=device) * 10
        return self.bank

    def generate_on_device(self, total_counts, device):
        # same random numbers, in the same order, as the numpy code in __call__
        ccsn_num = len(self.ccsn_list)
        ccsn_sample = np.random.choice(ccsn_num, total_counts)
        ccsn_counts = np.eye(ccsn_num)[ccsn_sample].sum(0).astype("int")
        index, theta, phi, distance = [], [], [], []
        for i, count in enumerate(ccsn_counts):
            index.append(np.full(count, i))
            theta.append(np.random.uniform(0, np.pi, count))
            phi.append(np.random.uniform(0, 2*np.pi, count))
            distance.append(np.random.uniform(1, 10, count))

        def to_device(x):
            return torch.as_tensor(np.concatenate(x), dtype=torch.float32, device=device)

        quad_moment = self.template_bank(device)[torch.as_tensor(np.concatenate(index), device=device)]
        hp, hc = get_hp_hc_from_q2ij(quad_moment, theta=to_device(theta), phi=to_device(phi), xp=torch)
        distance = to_device(distance).reshape(-1, 1)
        hp, hc = hp / distance, hc / distance

        dec = Cosine().rsample(total_counts)
        psi = Uniform(0, np.pi).sample((total_counts,))
        phi = Uniform(0, 2 * np.pi).sample((total_counts,))

        ht = gw.compute_observed_strain(
            dec.to(device),
            psi.to(device),
            phi.to(device),
            detector_tensors=self.tensors.to(device),
            detector_vertices=self.vertices.to(device),
            sample_rate=self.sample_rate,
            plus=hp,
            cross=hc
        )
        hrss = hrrs_value(hp, hc)

        return ht, dec, phi, hrss

    def __call__(
        self,
        total_counts, # batch_size
        device=None,
    ):
        if device is not None:
            return self.generate_on_device(total_counts, device)

        ccsn_num = len(self.ccsn_list)
        ccsn_sample = np.random.choice(ccsn_num, total_counts)
        ccsn_counts = np.eye(ccsn_num)[ccsn_sample].sum(0).astype("int")
        X = torch.empty((total_counts, 2, self.buffer_length))
        ccsne_agg_count = 0

        for name, count in zip(self.ccsn_list, ccsn_counts):

            time = self.signals[name][0]
            quad_moment = torch.Tensor(self.signals[name][1]) * 10

            theta = torch.Tensor(np.random.uniform(0, np.pi, count))
            phi = torch.Tensor(np.random.uniform(0, 2*np.pi, count))

            hp, hc = get_hp_hc_from_q2ij(
                quad_moment,
                theta=theta,
                phi=phi
            )

            hp_hc = padding(
                time,
                hp,
                hc,
                np.random.uniform(1, 10, count),
                sample_kernel = self.buffer_duration,
                sample_rate = self.sample_rate,
                time_shift = self.time_shift, # Core-bounce will be at here
            )

            X[ccsne_agg_count:ccsne_agg_count+count, :, :] = torch.Tensor(hp_hc)


            ccsne_agg_count += count
        # X = X[:, :, 2048:-2048]
        dec_distro = Cosine()
        psi_distro = Uniform(0, np.pi)
        phi_distro = Uniform(0, 2 * np.pi)

        dec = dec_distro.rsample(total_counts)
        psi = psi_distro.sample((total_counts,))
        phi = phi_distro.sample((total_counts,))

        ht = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=X[:,0,:],
            cross=X[:,1,:]
        )

        ht = ht.to('cuda') if torch.cuda.is_available() else ht
        hrss = hrrs_value(X[:,0,:], X[:,1,:])

        return ht, dec, phi, hrss
