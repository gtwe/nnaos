from typing import Tuple
import torch
import torch.distributions as tdist


class DataFun:
    """
    Base class for datasets generated from an explicit function.
    """

    def __init__(self, target_fn):

        if hasattr(self, 'data') and target_fn is not None:
            self.targets = target_fn(self.data)

    def __getitem__(self, i):

        return self.data[i], self.targets[i]

    def __len__(self):

        return len(self.data)


class DataCubeRandom(DataFun):
    """
    Random samples from a cube in :math:`\\verb|domain|^{\\verb|dim|}`.
    """

    def __init__(
        self,
        n_samples: int,
        dim: int = 1,
        target_fn=None,
        domain: Tuple[float, float] = (-1.0, 1.0),
    ):

        self.data = self.__class__.sample(n_samples, dim, domain)
        super().__init__(target_fn)

    @staticmethod
    def sample(n_samples, dim, domain):

        return tdist.uniform.Uniform(*domain).sample((n_samples, 1, dim))


class DataCubeGrid(DataFun):
    """
    Uniform grid on a cube in :math:`\\verb|domain|^{\\verb|dim|}`.
    """

    def __init__(
        self,
        n_samples_per_dim: int,
        dim: int = 1,
        target_fn=None,
        domain: Tuple[float, float] = (-1.0, 1.0),
    ):

        self.grid = self.__class__.make_grid(n_samples_per_dim, dim, domain)
        self.data = self.__class__.data_from_grid(self.grid)
        self.domain = domain
        super().__init__(target_fn)

    @staticmethod
    def make_grid(n_samples_per_dim, dim, domain):

        xi = torch.linspace(*domain, n_samples_per_dim)
        return torch.meshgrid(*(dim * [xi]))

    @staticmethod
    def data_from_grid(grid):

        dim = len(grid)
        return torch.stack(grid, dim=-1).reshape([-1, 1, dim])
