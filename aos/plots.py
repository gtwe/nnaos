import os
import numpy as np
import pandas as pd

# import scipy.stats as sstat
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import aos.data.data_fun as data


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


def loss_surface(df, expname, param, savedir=None, losstype=None):
    print(f'Plotting {losstype} loss surface against samples by DOF')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.view_init(30, 30)
    ax.tick_params(axis='x', labelrotation=-5, pad=0)
    ax.tick_params(axis='y', labelrotation=10)
    ax.tick_params(axis='z', pad=10)

    if param == "dof":
        ax.set_xlabel('DOF', labelpad=2, rotation=45)
        xticks = np.log(df.dof.unique())
        xtick_labels = df.dof.unique()
    elif param == "width":
        ax.set_xlabel('Width', labelpad=2, rotation=45)
        xticks = np.log(df.width.unique())
        xtick_labels = df.width.unique()

    ax.set_ylabel('Samples', labelpad=10)
    ax.set_zlabel('Loss', labelpad=13, rotation=90)

    yticks = np.log(df.samples.unique())

    color = "magma_r"

    if losstype == "Test":
        zticks = np.log(df.test_loss.unique())
        zscaling = np.linspace(
            np.log(df.test_loss.min()), np.log(df.test_loss.max()), 6
        )
        surf = ax.plot_trisurf(
            np.log(df.dof if param == "dof" else df.width),
            np.log(df.samples),
            np.log(df.test_loss),
            cmap=color,
            linewidth=10,
            antialiased=True,
        )
    elif losstype == "Train":
        zticks = np.log(df.loss.unique())
        zscaling = np.linspace(np.log(df.loss.min()), np.log(df.loss.max()), 6)
        surf = ax.plot_trisurf(
            np.log(df.dof if param == "dof" else df.width),
            np.log(df.samples),
            np.log(df.loss),
            cmap=color,
            linewidth=10,
            antialiased=True,
        )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zscaling)

    ax.set_yticklabels(df.samples.unique())
    ax.set_zticklabels([f"{np.exp(z): .3}" for z in zscaling])

    plt.savefig(
        f"{savedir}/{losstype}_surface_{param}.pdf", bbox_inches="tight", pad_inches=0.2
    )


def average_runs(df):
    df['test_loss'] = df.test_loss_history.apply(lambda x: x[-1])

    averaged_df_dof = (
        df.groupby(['id', 'dof', 'samples', 'dim', 'lr'])
        .agg({'loss': 'mean', 'test_loss': 'mean'})
        .reset_index()
    )

    averaged_df_width = (
        df.groupby(['id', 'width', 'samples', 'dim', 'lr'])
        .agg({'loss': 'mean', 'test_loss': 'mean'})
        .reset_index()
    )

    return averaged_df_dof, averaged_df_width


def large_loss_surface(df, expname, param, savedir=None, losstype=None):
    print(f'Plotting {losstype} loss surface against samples by {param}')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.view_init(30, 30)

    ax.set_ylabel('Samples', labelpad=10)
    ax.set_zlabel('Loss', labelpad=13, rotation=90)

    ax.tick_params(axis='x', labelrotation=-5, pad=0)
    ax.tick_params(axis='y', labelrotation=10)
    ax.tick_params(axis='z', pad=10)

    if param == "dof":
        xticks = np.log(df.dof.unique())
        ax.set_xlabel('DOF', labelpad=2, rotation=45)

    elif param == "width":
        xticks = np.log(df.width.unique())
        ax.set_xlabel('Width', labelpad=2, rotation=45)

    yticks = np.log(df.samples.unique())

    color = "magma_r"

    if losstype == "Test":
        zticks = np.log(df.test_loss.unique())
        zscaling = np.linspace(
            np.log(df.test_loss.min()), np.log(df.test_loss.max()), 6
        )
        surf = ax.plot_trisurf(
            np.log(df.width if param == "width" else df.dof),
            np.log(df.samples),
            np.log(df.test_loss),
            cmap=color,
            linewidth=10,
            antialiased=True,
        )
    elif losstype == "Train":
        zticks = np.log(df.loss.unique())
        zscaling = np.linspace(np.log(df.loss.min()), np.log(df.loss.max()), 6)
        surf = ax.plot_trisurf(
            np.log(df.width if param == "width" else df.dof),
            np.log(df.samples),
            np.log(df.loss),
            cmap=color,
            linewidth=10,
            antialiased=True,
        )

    xidx = [0, 4, 12, 28]
    yidx = [0, 4, 9, 14]

    ax.set_xticks(xticks[xidx])
    ax.set_yticks(yticks[yidx])
    ax.set_zticks(zscaling)

    xtickslabels = (
        df.width.unique()[xidx] if param == "width" else df.dof.unique()[xidx]
    )
    ytickslabels = df.samples.unique()[yidx]

    ax.set_xticklabels(xtickslabels)
    ax.set_yticklabels(ytickslabels)
    ax.set_zticklabels([f"{np.exp(z): .3}" for z in zscaling])

    plt.savefig(
        f"{savedir}/{losstype}_surface_{param}.pdf", bbox_inches="tight", pad_inches=0.2
    )


def exp_plotter(expname):
    df = pd.read_pickle("./results/df.pkl")
    savedir = f"../../reports/{expname}/plots"
    os.makedirs(savedir, exist_ok=True)
    averaged_df_dof, averaged_df_width = average_runs(df)

    if len(df.width.unique()) > 5:
        large_loss_surface(
            averaged_df_dof, expname, "dof", savedir=savedir, losstype="Train"
        )
        large_loss_surface(
            averaged_df_width, expname, "width", savedir=savedir, losstype="Train"
        )
        large_loss_surface(
            averaged_df_dof, expname, "dof", savedir=savedir, losstype="Test"
        )
        large_loss_surface(
            averaged_df_width, expname, "width", savedir=savedir, losstype="Test"
        )
    else:
        loss_surface(averaged_df_dof, expname, "dof", savedir=savedir, losstype="Train")
        loss_surface(
            averaged_df_width, expname, "width", savedir=savedir, losstype="Train"
        )
        loss_surface(averaged_df_dof, expname, "dof", savedir=savedir, losstype="Test")
        loss_surface(
            averaged_df_width, expname, "width", savedir=savedir, losstype="Test"
        )
