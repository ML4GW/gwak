import torch


def hrrs_value(
    h_plus: torch.Tensor,
    h_cross: torch.Tensor,
    dim: int = -1,
    dt: float = 1.0/4096,
):
    """
    Args:
        h_plus:  Tensor of shape (..., T)
        h_cross: Tensor of shape (..., T)
        dim:     Dimension over which to sum (default: last)
        dt:      Time interval between samples (default: 1.0/4096)

    Returns:
        Tensor of shape (...) with HRRS per batch element
    """
    # Squaring small strain values and scaling by dt can underflow in float32.
    # Use float64 for the calculation, then return the promoted input dtype.
    dtype = torch.promote_types(h_plus.dtype, h_cross.dtype)
    h_plus, h_cross = h_plus.double(), h_cross.double()
    hrrs = torch.sqrt(
        torch.sum((h_plus**2 + h_cross**2) * dt, dim=dim)
    )
    return hrrs.to(dtype)
