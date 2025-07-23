import torch
import time


class KernelLoss:

    def __init__(self, kernel_idx=1):
        self.kernel_idx = kernel_idx
        pass

    def __call__(self, pred, target):
        loss = torch.mean(pred - target, dim=self.kernel_idx)
        loss = torch.mean(loss**2)
        return loss


class GSELoss:

    def __init__(self):
        pass

    def set_target_fn(self, target_fn):
        self.target_fn = target_fn

    def set_network(self, network):
        self.network = network

    def set_x_and_y(self, x, y):
        self.x = x
        self.y = y

    def set_dim(self, dim):
        self.dim = dim

    def loss_fn(self, target, y):
        sigma = 0.001

        N = len(target)
        N_bar = 10

        samples = torch.randn(N_bar, self.dim) * sigma

        xmat = torch.empty(N_bar, N, self.dim)

        for i, xi in enumerate(self.x):
            col = samples + xi  # Math: \mathcal{N}(x_i, \sigma)
            xmat[:, i] = col

        # Math: \frac{1}{\bar{N}} \sum_{j=1}^\bar{N} \left[f_\theta(\mathcal{N}(x_i, \sigma)) - f(\mathcal{N}(x_i, \sigma))\right]^2
        # residual_by_i = 1 / N_bar * torch.sum(self.target_fn(xmat) - self.network(xmat), axis=0) ** 2
        residual_by_i = torch.mean(
            (self.target_fn(xmat) - self.network(xmat)) ** 2, axis=0
        )

        # assert residual_by_i.shape == (N,1)
        # Math: \frac{1}{N} \sum_{i=1}^N  \left[ \frac{1}{\bar{N}} \sum_{j=1}^\bar{N} \left[f_\theta(\mathcal{N}(x_i, \sigma)) - f(\mathcal{N}(x_i, \sigma))\right]^2\right]^2
        # gaussian_loss = 1 / N * torch.sum(residual_by_i ** 2)
        gaussian_loss = torch.mean(residual_by_i**2)

        return gaussian_loss
