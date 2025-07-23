import argparse

import yaml

import sys

sys.path.append("../../")

import aos.misc.time as ctime
import aos.callbacks as callbacks
import aos.models as models
import torch

from aos.misc.dicttools import CDict


class PrintSummary:
    def __init__(self, param):
        self.i_param = param.i_param
        self.n_params = param.n_params
        self.param = param

    def print(self):
        print(f"--- Experiment {self.i_param}/{self.n_params} -----------------------")
        print(f"id:            {self.param.id}")
        print(f"net_idx:       {self.param.net_idx}")
        print(f"epochs:        {self.param.epochs}")
        print(f"run:           {self.param.repetition}")
        print(f"train_target:  {self.param.train_dataset_args.target_fn}")
        print(f"train_loss:    {self.param.train_loss_fn.__name__}")
        print(f"test_target:   {self.param.test_dataset_args.target_fn}")
        print(f"test_loss:     {self.param.test_loss_fn.__name__}")
        print(f"dim:           {self.param.dim}")
        print(f"width:         {self.param.module_args.width}")
        print(f"depth:         {self.param.module_args.depth}")
        print(f"lr:            {self.param.optimizer_args.lr}")
        print(f"n_samples:     {self.param.train_dataset_args.n_samples}")
        print(f"test_samples:  {self.param.test_dataset_args.n_samples}")
        print(f"network_key:   {self.param.network_key}")
        print(f"tensor_type:   {self.param.tensor_type}")
        (
            print(f"random_seed:   {self.param.random_seed}")
            if self.param.random_seed
            else None
        )


def run(param):
    train_dataset = param.train_dataset(**param.train_dataset_args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        worker_init_fn=lambda _: (
            torch.manual_seed(param.random_seed) if param.random_seed else None
        ),
    )

    test_dataset = param.test_dataset(**param.test_dataset_args)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        worker_init_fn=lambda _: (
            torch.manual_seed(param.random_seed) if param.random_seed else None
        ),
    )

    network = param.module(**param.module_args)

    optimizer = param.optimizer(network.parameters(), **param.optimizer_args)

    model = models.Model(
        network, optimizer, param.train_loss_fn, param.test_loss_fn, epochs=param.epochs
    )

    cbs = [callbacks.PrintLoss(), callbacks.Log()]

    model.fit(train_loader, test_dataloader=test_loader, callbacks=cbs)

    return model


def main(paramloc):
    with open(paramloc, "r") as f:
        param = yaml.load(f, Loader=yaml.UnsafeLoader)

    print_summary = PrintSummary(param)
    print_summary.print()

    if param.random_seed:
        torch.manual_seed(param.random_seed)
        torch.use_deterministic_algorithms(True)

    dtype = param.tensor_type().dtype
    device = param.tensor_type().device

    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    with ctime.Timer() as time:
        model = run(param)
        result = CDict(
            {
                "params": param,
                "model": model,
            }
        )

    resultloc = paramloc.replace("params", "results")
    with open(resultloc, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paramloc", type=str)
    args = parser.parse_args()

    main(args.paramloc)
