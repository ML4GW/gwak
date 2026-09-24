import torch
import numpy as np

from scipy import signal
from ml4gw.transforms import SpectralDensity, Whiten


# precompute_embeddings
# fm_models
# combine_model
# plots
def frequency_cos_similarity(batch, mode):

    if mode == "random":
        rand_data = torch.randn(
            (batch.shape[0],),
            device=batch.device
        )
        return rand_data.unsqueeze(-1)

    if mode == "random2":
        rand_data = torch.randn(
            (batch.shape[0] * 2,),
            device=batch.device
        )
        return rand_data.reshape((-1, 2))

    H = torch.fft.rfft(batch[:, 0, :], dim=-1)
    L = torch.fft.rfft(batch[:, 1, :], dim=-1)
    numerator = torch.sum(H * torch.conj(L), dim=-1)
    norm_H = torch.linalg.norm(H, dim=-1)
    norm_L = torch.linalg.norm(L, dim=-1)
    rho_complex = numerator / (norm_H * norm_L + 1e-8)

    if mode == "real":
        return torch.real(rho_complex).unsqueeze(-1)
    if mode == "real_imag":
        score_1 = torch.real(rho_complex).unsqueeze(-1)
        score_2 = torch.imag(rho_complex).unsqueeze(-1)
        score = torch.cat([score_1, score_2], dim=1)
        return score
    if mode == "abs":
        # print(torch.abs(rho_complex).unsqueeze(-1).shape)
        return torch.abs(rho_complex).unsqueeze(-1)

    if mode == "half":
        H = torch.fft.rfft(batch[:, 0, 1024:-1024], dim=-1)
        L = torch.fft.rfft(batch[:, 1, 1024:-1024], dim=-1)
        numerator = torch.sum(H * torch.conj(L), dim=-1)
        norm_H = torch.linalg.norm(H, dim=-1)
        norm_L = torch.linalg.norm(L, dim=-1)
        rho_complex = numerator / (norm_H * norm_L + 1e-8)
        return torch.real(rho_complex).unsqueeze(-1)


class TorchBandpassFIR(torch.nn.Module):
    """
    Differentiable, zero-phase FIR band-pass filter.
    Works with [B, C, T] tensors.
    Can be TorchScripted for Triton deployment.
    """

    def __init__(
        self,
        highpass: float,
        lowpass: float,
        sample_rate: int = 4096,
        num_taps: int = 4096,
        zero_phase: bool = True,
    ):
        super().__init__()
        self.zero_phase = zero_phase

        # ---- FIR design ----
        fir_coeff = signal.firwin(
            numtaps=num_taps,
            cutoff=[highpass, lowpass],
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


class PreprocessBundle(torch.nn.Module):

    def __init__(
        self,
        sample_rate: int = 4096,
        fftlength: float = 2,
        fftaveraging: str = "median",
        highpass: float = 30,
        lowpass: float = 2047,
        padding_length: float = 2,
        psd_length: float = 64,
        minimal_kernel: float=1,
        full_return: bool=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.fftlength = fftlength
        self.fftaveraging = fftaveraging
        self.highpass = highpass
        self.lowpass = lowpass
        self.padding_length = padding_length
        self.psd_length = psd_length
        self.minimal_kernel = minimal_kernel
        self.full_return = full_return
        self.psd_samples = int(psd_length * sample_rate)
        self.padding_samples = int(padding_length * sample_rate)
        self.kernel_samples = int(minimal_kernel * sample_rate)
        self.required_samples = (
            self.psd_samples
            + self.padding_samples
            + self.kernel_samples
        )

        self.spectral_density = SpectralDensity(
            sample_rate=sample_rate,
            fftlength=fftlength,
            average=fftaveraging,
        )

        self.bandpass = TorchBandpassFIR(
            highpass=highpass,
            lowpass=lowpass,
            sample_rate=sample_rate,
        )

        self.whitener = Whiten(
            fduration=padding_length,
            sample_rate=sample_rate,
            highpass=highpass,
        )

    def forward(self, batch):
        data_length = batch.shape[-1]

        if data_length < self.required_samples:
            raise ValueError(
                f"Input data is too short: got {data_length} samples, "
                f"but at least {self.required_samples} samples are required."
            )
        psd_data = batch[...,:self.psd_samples]
        batch = batch[...,self.psd_samples:]

        batch = self.bandpass(batch)
        psds = self.spectral_density(psd_data.double())
        whitened = self.whitener(batch.double(), psds.double())

        if self.full_return:
            batch, psds, whitened
        return whitened