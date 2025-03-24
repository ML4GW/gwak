import h5py
import math
import torch

import numpy as np

from pathlib import Path
from ratelimiter import RateLimiter
from libs.time_slides import segments_from_paths, get_num_shifts_from_Tb
from deploy.libs.infer_utils import load_h5_as_dict, get_hp_hc_from_q2ij, on_grid_pol_to_sim, padding

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.utils.slicing import sample_kernels

def get_shifts_meta_data(
    background_fnames,
    Tb,
    shifts
):
    # calculate the number of shifts required
    # to accumulate the requested background,
    # given the duration of the background segments
    files, segments = segments_from_paths(background_fnames)
    num_shifts = get_num_shifts_from_Tb(
        segments, Tb, max(shifts)
    )
    return num_shifts, files, segments



class Sequence:

    def __init__(
        self,
        fname: Path,
        data_format: str, 
        shifts: list[float],
        batch_size: int,
        ifos: list,
        kernel_size: int,
        sample_rate: int,
        inference_sampling_rate: float,
        inj_type=None,
        precision: str="float32"
        # state_shape: tuple,
    ):

        # self.fname = fname
        self.shifts = shifts
        self.batch_size = batch_size
        self.ifos = ifos
        self.n_ifos = len(ifos)
        self.kernel_size = kernel_size

        self.state_shape = (self.n_ifos, kernel_size)

        self.inj_type = inj_type
        self.precision = precision

        self.sample_rate = sample_rate
        self.stride = int(sample_rate / inference_sampling_rate)
        self.step_size = self.stride * (kernel_size / sample_rate)
        
        self.strain_dict = {}
        self.fname = fname
        
        if data_format in ("h5", "hdf", "hdf5"):
            with h5py.File(self.fname, "r") as h:

                for ifo in self.ifos:
                    self.strain_dict[ifo] = h[ifo][:].astype(self.precision)

        self.size = len(self.strain_dict[ifo])

    @property
    def remainder(self):
        # the number of remaining data points not filling a full batch
        return (self.size - max(self.shifts)) % self.step_size

    @property
    def num_pad(self):
        # the number of zeros we need to pad the last batch
        # to make it a full batch
        return int((self.step_size - self.remainder) % self.step_size)
    
    def __len__(self):

        return math.ceil((self.size - max(self.shifts)) / self.step_size)

    def __iter__(self):

        # Check if this line will hide potential implmetation error! 
        limiter = RateLimiter(max_calls=1, period=0.1)
        bg_state = np.empty(self.state_shape, dtype=self.precision) 
        inj_state = None

        for i in range(len(self)):

            last = i == len(self) - 1
            for ifo_idx, (ifo, shift) in enumerate(zip(self.ifos, self.shifts)): 

                start = int(shift + i * self.step_size)
                end = int(start + self.kernel_size)


                if last and self.remainder:
                    end = start + int(self.remainder)

                data = self.strain_dict[ifo][start:end]
                # if this is the last batch
                # possibly pad it to make it a full batch
                if last:

                    data = np.pad(data, (0, self.num_pad), "constant")


                bg_state[ifo_idx, :] = data

            
            inj_state = bg_state
            
            with limiter:
                yield bg_state, inj_state





    
class CCSN_Waveform_Projector: 

    def __init__(
        self,
        ifos,
        sample_rate,
        sample_duration,
        buffer_duration=3,
        time_shift=0,
        off_set=0,
    ):  
        """_summary_

        Args:
            ifos (_type_): _description_
            sample_rate (_type_): _description_
            sample_duration (_type_): _description_
            buffer_duration (int, optional): _description_. Defaults to 3.
            time_shift (int, optional): Shifts Core-bounce to "Time Shift". Defaults to 0.
            off_set (int, optional): _description_. Defaults to 0.
        """


        self.ifos = ifos
        self.sample_rate = sample_rate
            
        self.tensors, self.vertices = gw.get_ifo_geometry(*self.ifos)

        self.sample_duration = sample_duration

        self.buffer_duration = buffer_duration

        self.off_set = off_set
        self.time_shift = time_shift

        if off_set is not None:
            
            self.max_center_offset = int((buffer_duration/2 - sample_duration - off_set) * sample_rate)

    def __call__(
        self,
        time,
        quad_moment,
        ori_theta,
        ori_phi,
        dec,
        psi,
        phi,
        default_snr: float = 4
    ):
        
        count = len(ori_theta)
        
        hp, hc = get_hp_hc_from_q2ij(
            quad_moment,
            theta=ori_theta,
            phi=ori_phi
        )

        hp_hc = padding(
            time,
            hp,
            hc,
            np.ones(count),
            sample_kernel = self.buffer_duration,
            sample_rate = self.sample_rate,
            time_shift = self.time_shift, # Core-bounce will be at here
        )

        hp_hc = torch.tensor(hp_hc).float()
        
        if self.buffer_duration > self.sample_duration:
        
            hp_hc = sample_kernels(
                X = hp_hc,  
                kernel_size = self.sample_rate * self.sample_duration,
                max_center_offset = self.max_center_offset,
            )

        ht = gw.compute_observed_strain(
            torch.tensor(dec).float(),
            torch.tensor(psi).float(),
            torch.tensor(phi).float(),
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=hp_hc[:,0,:],
            cross=hp_hc[:,1,:]
        )

        return ht.detach().numpy()
        
        