from torch.nn import MSELoss
from torch.optim import SGD
from torch import FloatTensor, DoubleTensor

import sys
import os

import itertools

import random
import string

import yaml
import glob

sys.path.append("../../")

from aos.misc.dicttools import Eval, Product, product_idx, resolve
from aos.data.data_fun import DataCubeRandom, KernelDataCubeRandom
from aos.data.functions import CuspNd, GaussianNd
from aos.nets import FullyConnected


class Experiment:
    def __init__(self):
        self.expname = "dim7dep2_big_batch_mse"
        self.nets = [
            {
                "name": "wide_shallow",
                "module": FullyConnected,
                "module_args": {
                    "width": Product([_ for _ in range(32, 257, 8)], group="dof"),
                    "depth": 2,
                    "in_dim": Eval(lambda r: r.dim),
                },
                "net_idx": product_idx(group="dof"),  # Ordering for rate calculations
            },
        ]
        self.common = {
            "train_dataset": DataCubeRandom,
            "train_dataset_args": {
                "n_samples": Product(
                    [_ for _ in range(50, 751, 50)], group="n_samples"
                ),
                "dim": Eval(lambda r: r.dim),
                "target_fn": Eval(lambda r: GaussianNd(r.dim)),
            },
            "train_loss_fn": MSELoss,
            "test_dataset": DataCubeRandom,
            "test_dataset_args": {
                "n_samples": 1000,
                "dim": Eval(lambda r: r.dim),
                "target_fn": Eval(lambda r: GaussianNd(r.dim)),
            },
            "test_loss_fn": MSELoss,
            "optimizer": SGD,
            "optimizer_args": {
                "lr": Product([0.05], group="lr"),
            },
            "epochs": Product([20000], group="epochs"),
            "dim": Product([7], group="dim"),
            "repetition": Product(list(range(20)), group="repeat"),
            "id": Eval(lambda r: f"{r.name}"),
            "n_params": None,
            "i_param": None,
            "random_seed": None,  # Set to None, or a specific seed
            "tensor_type": FloatTensor,
        }

    def get_params(self):
        return self.nets, self.common

    def get_expname(self):
        return self.expname

    def set_expname(self, expname):
        self.expname = expname


def mkdirs(expname):
    directories = ["params", "results"]

    for dir in directories:
        os.makedirs(f"{dir}", exist_ok=True)


def writeparams(expname, nets, common, loc=None):
    params = list(itertools.chain(*[resolve({**net, **common}) for net in nets]))

    n_params = len(params)

    for i, param in enumerate(params):
        param["i_param"] = i + 1
        param["n_params"] = n_params
        param["network_key"] = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=8)
        )

    n_digits = len(str(n_params))

    exp_nums = []

    files = glob.glob(f"./params/*.yaml")
    for file in files:
        print(f"Removing {file}")
        os.remove(file)

    files = glob.glob(f"./results/*.yaml")
    for file in files:
        print(f"Removing {file}")
        os.remove(file)

    for i, param in enumerate(params):
        pad_count = str(i + 1).zfill(n_digits)
        exp_nums.append(pad_count)
        run_params_loc = f"./params/exp_{pad_count}.yaml"
        with open(run_params_loc, "w") as file:
            yaml.dump(param, file, default_flow_style=False, sort_keys=False)

    return exp_nums


if __name__ == "__main__":
    exp = Experiment()
    expname = exp.get_expname()
    nets, common = exp.get_params()
    print(f"Writing params for {expname}")
    mkdirs(expname)
    exp_nums = writeparams(expname, nets, common)
