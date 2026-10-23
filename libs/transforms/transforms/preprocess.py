import torch
import numpy as np

from scipy import signal
from ml4gw.transforms import SpectralDensity, Whiten

# from .correlation import FEATURES as CORRELATION_MODES, correlation_features
from .correlation import FEATURES
from .correlation import CorrelationFeatures

from .arch_conversion import cohmode_to_dim

# precompute_embeddings
# fm_models
# combine_model
# plots

class FeatureExtractor(torch.nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode in FrequencyCosSimilarity.MODES:
            self.feature = FrequencyCosSimilarity(mode=mode)
        if mode in FEATURES.keys():
            self.feature = CorrelationFeatures(mode=mode)

    
    def forward(self, batch):
        return self.feature(batch)


class FrequencyCosSimilarity(torch.nn.Module):
    """
    Complex cosine similarity between the spectra of two channels.

    Expects `batch` of shape (B, 2, T): channel 0 is H, channel 1 is L.
    Returns a tensor of shape (B, out_dim).
    """

    MODES = ("random", "random2", "real", "real_imag", "abs", "half")

    def __init__(
        self,
        mode: str = "real",
        eps: float = 1e-8,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.eps = eps

        _call_fn = {
            "random": self._random,
            "random2": self._random2,
            "real": self._real,
            "real_imag": self._real_imag,
            "abs": self._abs,
            "half": self._half
        }

        self._foward_fn = _call_fn.get(mode)

    @property
    def out_dim(self) -> int:
        return cohmode_to_dim(self.mode)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._foward_fn(batch)

    # Testing use
    def _random(self, batch: torch.Tensor) -> torch.Tensor:
        n = batch.shape[0]
        return torch.randn((n, 1), device=batch.device)

    def _random2(self, batch: torch.Tensor) -> torch.Tensor:
        n = batch.shape[0]
        return torch.randn((n, 2), device=batch.device)

    def _complex_similarity(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        H = torch.fft.rfft(batch[:, 0, :], dim=-1)
        L = torch.fft.rfft(batch[:, 1, :], dim=-1)
        numerator = torch.sum(H * torch.conj(L), dim=-1)
        norm_H = torch.linalg.norm(H, dim=-1)
        norm_L = torch.linalg.norm(L, dim=-1)
        rho_complex = numerator / (norm_H * norm_L + 1e-8)

        return rho_complex

    def _real(self, batch: torch.Tensor) -> torch.Tensor:
        rho_complex = self._complex_similarity(batch)
        return torch.real(rho_complex).unsqueeze(-1)

    def _real_imag(self, batch: torch.Tensor) -> torch.Tensor:
        rho_complex = self._complex_similarity(batch)
        score_1 = torch.real(rho_complex).unsqueeze(-1)
        score_2 = torch.imag(rho_complex).unsqueeze(-1)
        score = torch.cat([score_1, score_2], dim=1)
        return score

    def _abs(self, batch: torch.Tensor) -> torch.Tensor:
        rho_complex = self._complex_similarity(batch)
        return torch.abs(rho_complex).unsqueeze(-1)

    def _half(self, batch: torch.Tensor) -> torch.Tensor:
        """Half window size of real _complex_similarity"""
        batch = batch[..., 1024:-1024]
        rho_complex = self._complex_similarity(batch)
        return torch.real(rho_complex).unsqueeze(-1)


def frequency_cos_similarity(batch, mode):

    if mode in CORRELATION_MODES:
        # H1-L1 correlation features (transforms/correlation.py): (B, 15) or (B, 21)
        return correlation_features(batch, mode=mode)

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