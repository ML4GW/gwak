
import torch
import h5py
from pathlib import Path

from ml4gw.transforms import SpectralDensity, Whiten

from background_utils import Sequence


def generate_timeslide(
    fduration = 1,
    sample_rate = 2048,
    highpass = 30,
    fftlength = 2,
    kernel_length = 0.09765625,  # 0.09765625
    psd_length = 64,
    data_dir = Path("/home/katya.govorkova/gwak2_background"),
    whitened_dir = Path("/home/hongyin.chen/whiten_timeslide_short.h5"),
    data_format = "hdf5",
    shifts = 2,
    ifos = ["H1", "L1"],
    device="cuda"
):

    kernel_size = int((psd_length + kernel_length + fduration) * sample_rate)

    whitener = Whiten(fduration, sample_rate, highpass).to(device)
    spectral_density = SpectralDensity(sample_rate,fftlength,average='median').to(device)

    split_size = int((kernel_length + fduration) * sample_rate)
    splits = [psd_length * sample_rate, split_size]

    whitened_data = []
    
    for shift in range(1, shifts + 1):
        for strain_file in data_dir.glob(f"*.{data_format}"):
            
            sequence = Sequence(
                fname=strain_file,
                data_format=data_format,
                shifts=[0, shift],
                # batch_size=batch_size,0
                ifos=ifos,
                kernel_size=kernel_size,
                sample_rate=sample_rate,
                # inference_sampling_rate=inference_sampling_rate,
                inj_type=None,
            )
            print(shift)
            for idx, (bh_state, _) in enumerate(sequence):
                if idx == (len(sequence) - 1):
                    continue
                bh_state = torch.Tensor(bh_state).to(device)
                # breakpoint()
                psd_data, batch = torch.split(bh_state, splits, dim=-1)
                psds = spectral_density(psd_data.double())
                
                whitened = whitener(batch.double(), psds.double())

                whitened_data.append(whitened)

    whitened_data = torch.vstack(whitened_data)
    whitened_data = whitened_data.cpu().detach().numpy()
    with h5py.File(whitened_dir, "w") as g:

        g.create_dataset("data", data=whitened_data)
        
        
if __name__ == "__main__":
    
    generate_timeslide()