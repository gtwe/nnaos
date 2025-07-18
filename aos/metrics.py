import itertools
import operator
import typing
import numpy as np
import torch


def dof(module: torch.nn.Module) -> int:
    """
    Number of trainable weights in a neural network.
    """
    return sum([p.numel() for p in module.parameters() if p.requires_grad])


def rates(dofs: typing.Iterable, errors: typing.Iterable) -> list:
    """
    Numerical estimate of convergence rates.
    """
    assert len(dofs) == len(errors)

    r = (len(dofs) - 1) * [None]
    for i in range(len(dofs) - 1):
        r[i] = -np.log(errors[i + 1] / errors[i]) / np.log(dofs[i + 1] / dofs[i])

    return r


def add_rates(
    logs: typing.List[dict],
    split_keys: typing.Tuple[typing.Hashable] = (),
    dof_key: typing.Hashable = 'dof',
    loss_key: typing.Hashable = 'loss',
    rate_key: typing.Hashable = 'rate',
) -> typing.List[dict]:
    """
    Compute numerical rate for a list of dictionaries.

    The log dictionaries may come from severl independent experiments,
    which can be identified by `split_key`.

    Args:
        logs: List of dictionaries `log` containing all necessary data.
        split_key: If the `log` contain multiple unrelated experiments,
            they can be split different `split_keys`.
        dof_key: `log` key containing the degrees of freedom.
        loss_key: `log` key containing the losses.
        rate_key: Rates are added to `log` with this key.

    Returns:
        `logs` with added rates in key `rate_key`.
        The method may change the ordering of `logs`.
    """

    # The log dicts are sorted by experiment type in `split_key` and
    # then by `dof_key`, so that we can determine rates oftwo
    # consecutive logs.

    logs.sort(key=lambda s: operator.itemgetter(*split_keys, dof_key)(s))

    # logs with same `group_fn` belog to the same group of experiments.

    if split_keys != ():

        def group_fn(s):
            return operator.itemgetter(*split_keys)(s)

    else:

        def group_fn(s):
            return None

    # Compute a rate from two consecutive logs, except when they are
    # not from the same group, i.e. have different `group_fn`.

    logs[0][rate_key] = np.nan
    for i, s in itertools.islice(enumerate(logs), 1, None):
        s_prev = logs[i - 1]
        if group_fn(s) == group_fn(s_prev):
            dofs = [s_prev[dof_key], s[dof_key]]
            losses = [np.mean(s_prev[loss_key]), np.mean(s[loss_key])]
            s[rate_key] = rates(dofs, losses)[0]
        else:
            s[rate_key] = np.nan

    return logs
