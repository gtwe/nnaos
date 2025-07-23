import pandas as pd
import aos.metrics as mtx
import numpy as np

import git


def create_frame(experiments: list) -> pd.DataFrame:
    """
    Purposed to take in a list and return a DataFrame based on the input list.

    """

    df = pd.DataFrame(
        columns=[
            "id",
            "network_key",
            "dof",
            "samples",
            "test_samples",
            "dim",
            "lr",
            "run",
            "loss_history",
            "test_loss_history",
            "test_loss",
            "loss",
            "model",
            "params",
            "width",
        ]
    )

    for e in experiments:
        # can't use df.loc[-1]
        df.loc[len(df)] = {
            "id": e.params.id,
            "network_key": e.params["network_key"],
            "dof": e.model.log["dof"],
            "samples": e.params.train_dataset_args["n_samples"],
            "test_samples": e.params.test_dataset_args["n_samples"],
            "run": e.params["repetition"],
            "dim": e.params["dim"],
            "lr": e.params.optimizer_args["lr"],
            "loss_history": e.model.log["train/loss"],
            "test_loss_history": e.model.log["test/loss"],
            "test_loss": e.model.log["test/loss"][-1],
            "loss": e.model.log["train/loss"][-1],
            "model": e.model,
            "params": e.params,
            "width": e.params.module_args.width,
        }

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    df["Git Hash"] = sha
    return df


def add_rate_col(
    dof_key: str = "dof", loss_key: str = "loss", rate_key: str = "rate"
) -> object:
    """
    Purposed to sort a Dataframe by degrees of freedom, compute rates, and insert those
    rates into the DataFrame.

    Returns a Dataframe with 'rate' column appended.

    """

    def _add_rate_col(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(dof_key)

        rates = mtx.rates(df[dof_key].to_numpy(), df[loss_key].to_numpy())
        rates.insert(0, np.nan)

        df[rate_key] = rates
        return df

    return _add_rate_col


# tmp = add_rate_col(dof_key = "width", ...)
# df.groupby(...).apply(tmp)
# df = tmp(df)
