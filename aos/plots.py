import numpy as np

# import scipy.stats as sstat
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sparse.data.data_fun as data


def tnp(tensor):

    return tensor.detach().numpy()


def plot_breakpoints_1d(ax, module, target_fn, domain=(-1, 1), margin=0.5):
    """
    Plot the location of the discontinuities of the first layer.

    The figure includes three parts:

    - A plot of the target function.
    - The neural network approximation.
    - The breakpoints of the first layer (x-axis), normalized outer layer weight (y-axis).
    - A histogram of the breakpoints.
    """
    n_samples_per_dim = 50

    weight = module.layers[0].weight
    bias = module.layers[0].bias[:, None]
    a = torch.flatten(0.1 * F.normalize(module.layers[-1].weight, p=np.inf))
    breakpoints = torch.flatten(-bias / weight)

    dim = weight.shape[-1]
    assert dim == 1, f'Error: dim must be 1 but is {dim}.'

    grid = data.DataCubeGrid.make_grid(n_samples_per_dim, dim=dim, domain=domain)
    x = data.DataCubeGrid.data_from_grid(grid)
    y = torch.flatten(module(x))
    z = torch.squeeze(target_fn(torch.stack(grid, dim=-1)))

    ax.set_xlim(domain[0] - margin, domain[1] + margin)
    bins = int(breakpoints.nelement() / 10)

    ax.hist(tnp(breakpoints), bins=bins, density=True, color='slateblue', alpha=0.2)
    ax.scatter(tnp(breakpoints), tnp(a), color='slateblue')
    ax.plot(tnp(grid[0]), tnp(z), color='tomato', alpha=0.5)
    ax.plot(tnp(grid[0]), tnp(y), color='royalblue')

    # # Plot an estimated probability density of the biases.
    # active_breakpoints = breakpoints[torch.abs(breakpoints) > 1e-2]
    # kde = sstat.gaussian_kde(tnp(active_breakpoints))
    # ax.plot(tnp(grid[0]), kde(tnp(grid[0])), color='slateblue')


def plot_breakpoints_2d(ax, module, target_fn, domain=(-1, 1), margin=0.5):
    """
    Plot the location of the discontinuities of the first layer.

    The figure includes three parts:

    - A `contourf` plot of the target function.
    - For each ReLU in the first layer, the point on its discontinuity
      line that is closes to the origin.
    - The origin in a different color.
    """
    n_samples_per_dim = 50

    weight = module.layers[0].weight
    bias = module.layers[0].bias[:, None]
    dim = weight.shape[-1]

    assert dim == 2, f'Error: dim must be 2 but is {dim}.'

    # Point on the ReLU kink hyperplane closes to the origin
    x = -bias / torch.norm(weight, dim=1, keepdim=True) ** 2 * weight

    grid = data.DataCubeGrid.make_grid(n_samples_per_dim, dim=dim, domain=domain)
    z = torch.squeeze(target_fn(torch.stack(grid, dim=-1)))

    ax.set_xlim(domain[0] - margin, domain[1] + margin)
    ax.set_ylim(domain[0] - margin, domain[1] + margin)
    ax.contourf(tnp(grid[0]), tnp(grid[1]), tnp(z))
    ax.scatter(*x.detach().numpy().T)
    ax.scatter(0, 0, color='red')


def plot_breakpoints(module, target_fn, domain=(-1, 1)):

    dim = module.layers[0].weight.shape[-1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if dim == 1:
        plot_breakpoints_1d(ax, module, target_fn, domain)
    elif dim == 2:
        plot_breakpoints_2d(ax, module, target_fn, domain)

    return fig
