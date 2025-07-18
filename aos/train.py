import typing
import torch
import torch.utils.data


def train_loop(
    network: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], None],
    callback_fn: typing.Callable[[dict], None],
) -> None:
    """
    Training loop for one epoch.
    """
    for batch, (x, y) in enumerate(dataloader):

        pred = network(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        callback_fn((x, y), {'loss': loss, 'pred': pred})


def test_loop(
    network: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    callback_fn: typing.Callable[[dict], None],
) -> None:
    """
    Test loop for one epoch.
    """
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            pred = network(x)
            callback_fn((x, y), {'pred': pred})
