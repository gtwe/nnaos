import typing
import torch
import torch.utils.data


def train_loop(
    network: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    train_loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], None],
    callback_fn: typing.Callable[[dict], None],
) -> None:
    """
    Training loop for one epoch.
    """
    for batch, (x, y) in enumerate(dataloader):

        pred = network(x)
        if len(x.shape) == 3:
            # Usual case
            # (n_samples, 1, dim)
            loss = train_loss_fn()(pred, y)
        if len(x.shape) == 4:
            # Kernel loss
            # (n_samples, 1, dim, kernel_size)
            loss = torch.mean(
                pred - y, dim=1
            )  # average residual across column in kernel
            loss = torch.mean(loss**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        callback_fn((x, y), {'loss': loss, 'pred': pred})


def test_loop(
    network: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    test_loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], None],
    callback_fn: typing.Callable[[dict], None],
) -> None:
    """
    Test loop for one epoch.
    """
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            pred = network(x)
            loss = test_loss_fn()(pred, y)
            callback_fn((x, y), {'loss': loss, 'pred': pred})
