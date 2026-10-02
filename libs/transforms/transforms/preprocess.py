import torch
import numpy as np

from scipy import signal

# precompute_embeddings
def frequency_cos_similarity(batch, mode):
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
        return torch.abs(rho_complex).unsqueeze(-1)


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


# fm_models
# combine_model
# plots

