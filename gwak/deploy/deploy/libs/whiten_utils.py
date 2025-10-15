import torch 

import numpy as np
from scipy import signal
from typing import Callable, Optional, Tuple

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows

Tensor = torch.Tensor

class TorchBandpassFIR(torch.nn.Module):
    """
    Differentiable, zero-phase FIR band-pass filter.
    Works with [B, C, T] tensors.
    Can be TorchScripted for Triton deployment.
    """

    def __init__(
        self,
        lowcut: float,
        highcut: float,
        sample_rate: int = 4096,
        # order: int = 8,
        num_taps: int = 4096,
        zero_phase: bool = True,
    ):
        super().__init__()
        self.zero_phase = zero_phase

        # ---- FIR design ----
        fir_coeff = signal.firwin(
            numtaps=num_taps,
            cutoff=[lowcut, highcut],
            pass_zero=False,
            fs=sample_rate,
        ).astype(np.float32)

        # Conv1d expects [out_channels, in_channels/groups, kernel]
        kernel = torch.tensor(fir_coeff, dtype=torch.float32).view(1, 1, -1)

        self.register_buffer("kernel", kernel)
        self.pad = num_taps // 2
        self.num_taps = num_taps

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal grouped convolution.
        x: [B, C, T]
        """
        B, C, T = x.shape
        weight = self.kernel.repeat(C, 1, 1)  # one kernel per channel
        return torch.nn.functional.conv1d(x, weight, groups=C, padding="same")

    def forward(self, x):
        pad = self.num_taps // 2
        x_pad = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
        y = self._conv(x_pad)
        if self.zero_phase:
            y = torch.flip(y, dims=[-1])
            y = self._conv(y)
            y = torch.flip(y, dims=[-1])
        return y[..., pad:-pad]

class BackgroundSnapshotter(torch.nn.Module):
    """Update a kernel with a new piece of streaming data"""

    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        super().__init__()
        state_length = kernel_length + fduration + psd_length
        state_length -= 1 / inference_sampling_rate
        self.state_size = int(state_length * sample_rate)

    def forward(self, update: Tensor, snapshot: Tensor) -> Tuple[Tensor, ...]:
        x = torch.cat([snapshot, update], axis=-1)
        snapshot = x[:, :, -self.state_size :]
        return x, snapshot


class PsdEstimator(torch.nn.Module):
    """
    Module that takes a sample of data, splits it into
    two unequal-length segments, calculates the PSD of
    the first section, then returns this PSD along with
    the second section.

    Args:
        length:
            The length, in seconds, of timeseries data
            to be returned for whitening. Note that the
            length of time used for the PSD will then be
            whatever remains along first part of the time
            axis of the input.
        sample_rate:
            Rate at which input data has been sampled in Hz
        fftlength:
            Length of FFTs to use when computing the PSD
        overlap:
            Amount of overlap between FFT windows when
            computing the PSD. Default value of `None`
            uses `fftlength / 2`
        average:
            Method for aggregating spectra from FFT
            windows, either `"mean"` or `"median"`
        fast:
            If `True`, use a slightly faster PSD algorithm
            that is inaccurate for the lowest two frequency
            bins. If you plan on highpassing later, this
            should be fine.
    """

    def __init__(
        self,
        length: int,
        sample_rate: float,
        fftlength: float,
        window: Optional[torch.Tensor] = None,
        overlap: Optional[float] = None,
        average: str = "median",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        self.spectral_density = SpectralDensity(
            sample_rate, 
            fftlength, 
            overlap, 
            average, 
            window=window, 
            fast=fast
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        splits = [X.size(-1) - self.size, self.size] # num_info at axis=-1
        background, X = torch.split(X, splits, dim=-1)

        # if we have 2 batch elements in our input data,
        # it will be assumed that the 0th element is data
        # being used to calculate the psd to whiten the
        # 1st element. Used when we want to use raw background
        # data to calculate the PSDs to whiten data with injected signals
        if X.ndim == 3 and X.size(0) == 2:
            # 0th background element is used to calculate PSDs
            background = background[0]
            # 1st element is the data to be whitened
            X = X[-1] # Strain data is at axis=-1

        psds = self.spectral_density(background.double())
        return X, psds



class BatchWhitener(torch.nn.Module):
    """Calculate the PSDs and whiten an entire batch of kernels at once"""

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        augmentor: Optional[Callable] = None,
        highpass: Optional[float] = None,
        return_whitened: bool = False,
    ) -> None:
        super().__init__()
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.augmentor = augmentor
        self.return_whitened = return_whitened

        # do foreground length calculation in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate
        
        self.bandpass = TorchBandpassFIR(
                    lowcut=highpass,
                    highcut=2047,
                    sample_rate=sample_rate
        )
        
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            # fast=highpass is not None,
            fast=False
        )
        self.whitener = Whiten(fduration, sample_rate, highpass)

    def forward(self, x: Tensor) -> Tensor:

        # Get the number of channels so we know how to
        # reshape `x` appropriately after unfolding to
        # ensure we have (batch, channels, time) shape
        if x.ndim == 3:
            num_channels = x.size(1)
        elif x.ndim == 2:
            num_channels = x.size(0)
        else:
            raise ValueError(
                "Expected input to be either 2 or 3 dimensional, "
                "but found shape {}".format(x.shape)
            )
        x, psd = self.psd_estimator(x)
        x = x.reshape(-1, num_channels, x.size(-1))
        x = self.bandpass(x)
        x = x.reshape(num_channels, -1)
        whitened = self.whitener(x.double(), psd.double())

        # unfold x and then put it into the expected shape.
        # Note that if x has both signal and background
        # batch elements, they will be interleaved along
        # the batch dimension after unfolding
        # x = unfold_windows(whitened, self.kernel_size, self.stride_size) # (batch, num_ifo, kernel_size)
        # x = x.reshape(-1, num_channels, self.kernel_size) # Apply this for gwak_1
        # x = x.reshape(-1, self.kernel_size, num_channels) # Apply this for gwak_1
        # whitened = whitened.transpose(1, 2) # Apply this for gwak_1
        x = unfold_windows(whitened, self.kernel_size, self.stride_size)
        x = x.reshape(-1, num_channels, self.kernel_size)
        stds = torch.std(x, dim=-1, keepdim=True)
        x = x / stds
        if self.augmentor is not None:
            x = self.augmentor(x)
        if self.return_whitened:
            return x, whitened
        return x