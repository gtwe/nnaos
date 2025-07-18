import torch


def cusp_nd(dim: int):
    """
    Returns a cusp function :math:`x \\to \\sqrt{|w^T x|}` for random :math:`w` in :math:`n` dimensions.

    The input :math:`x` can be batched with size `dim` in the last axis.
    """

    w = torch.normal(mean=0.0, std=1.0, size=(dim,))
    w /= torch.linalg.norm(w)

    def _cusp_nd(x: torch.Tensor) -> torch.Tensor:

        return torch.sqrt(torch.abs(torch.sum(x * w, dim=-1, keepdim=True)))

    return _cusp_nd
