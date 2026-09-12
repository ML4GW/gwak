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
    hrrs = torch.sqrt(
        torch.sum((h_plus**2 + h_cross**2) * dt, dim=dim)
    )
    return hrrs