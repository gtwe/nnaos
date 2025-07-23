import torch


class CuspNd:
    """
    Returns a cusp function :math:`x \\to \\sqrt{|w^T x|}` for random :math:`w` in :math:`n` dimensions.

    The input :math:`x` can be batched with size `dim` in the last axis.
    """

    def __init__(self, dim: int):
        self.w = torch.normal(mean=0.0, std=1.0, size=(dim,))
        self.w /= torch.linalg.norm(self.w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.abs(torch.sum(x * self.w, dim=-1, keepdim=True)))


class GaussianNd:
    def __init__(self, dim: int):
        pass

    def __call__(self, x: torch.Tensor):
        return torch.exp(-0.5 * torch.sum(x * x, dim=-1, keepdim=True))
