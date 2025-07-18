import math
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """
    Standard dense fully connected neural network.

    """

    def __init__(
        self, width: int, depth: int, in_dim: int = 1, out_dim: int = 1
    ) -> None:

        assert depth >= 2, "At least depth 2 required."

        super().__init__()

        self.activation = nn.ReLU

        start = []
        self.layers = nn.Sequential(
            *sum(
                [
                    [nn.Linear(in_dim, width), self.activation()],
                    *[
                        [nn.Linear(width, width), self.activation()]
                        for _ in range(depth - 2)
                    ],
                    [nn.Linear(width, out_dim, bias=False)],
                ],
                start,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layers(x)
