import h5py
import math

import numpy as np

from pathlib import Path
from libs.time_slides import segments_from_paths, get_num_shifts_from_Tb


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

        self.state_shape = (batch_size, self.n_ifos, kernel_size)

        self.inj_type = inj_type
        self.precision = precision

        self.sample_rate = sample_rate
        self.stride = int(sample_rate / inference_sampling_rate)
        self.step_size = self.stride * (kernel_size / sample_rate)
        
        self.strain_dict = {}
        self.fname = fname
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
        bh_state = np.empty(self.state_shape, dtype=self.precision) 
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


                bh_state[:, ifo_idx, :] = data
                
                # if end == self.rate_limit:
                #     break


            if self.inj_type is not None:
                inj_state = self.bh_state + "signal"
                        
            yield bh_state, inj_state
