import typing
import numpy as np

from . import log
from . import metrics


class Callback:
    """
    Call back function for training.

    The implementation is similar to callbacks in
    Keras and Lightning.
    """

    def set_model(self, model):
        self.model = model

    def set_data(self, train_dataloader, test_dataloader=None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def on_train_batch_end(self, batch, outputs):

        pass

    def on_train_epoch_end(self, epoch):

        pass

    def on_train_epoch_start(self, epoch):

        pass

    def on_test_batch_end(self, batch, output):

        pass

    def on_test_epoch_end(self, epoch):
        pass

    def on_experiment_end(self, epochs):
        pass


def callback_fn(
    callbacks: typing.List[Callback],
) -> typing.Callable[[str], typing.Callable[..., None]]:
    """
    Turn a list of callbacks into a single callback function.
    """

    def _callback(method):
        def _callback_fn(*args, **kwargs):
            for c in callbacks:
                getattr(c, method)(*args, **kwargs)

        return _callback_fn

    return _callback


class PrintLoss(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.n_print = 20

    def set_model(self, model):
        self.epochs = model.epochs

    def on_train_batch_end(self, batch, output):
        self.loss = output['loss'].item()

    def on_test_batch_end(self, batch, output):
        self.test_loss = output['loss'].item()

    # def on_train_epoch_end(self, epoch):
    # The max prevents division by zero for small num_batches.
    # if epoch % max(1, (self.epochs // self.n_print)) == 0:
    # print(f"Epoch {epoch:>5d}, Loss {self.loss:>7f}, logLoss {np.log(self.loss):>7f}")

    def on_test_epoch_end(self, epoch):
        if epoch % max(1, (self.epochs // self.n_print)) == 0:
            print(
                f"Epoch {epoch:>5d}, Train Loss {self.loss:>7f}, Test Loss {self.test_loss:>7f}"
            )

    def on_experiment_end(self, epochs):
        pass


class Log(Callback):

    def set_model(self, model):
        super().set_model(model)
        self.model.log = log.Logger()
        self.model.log['dof'] = metrics.dof(model.network)

    def on_train_batch_end(self, batch, outputs):
        self.model.log.append('train/loss', outputs['loss'].item())

    def on_test_batch_end(self, batch, outputs):
        self.model.log.append('test/loss', outputs['loss'].item())

    def on_experiment_end(self, epochs):
        idx = np.round(np.linspace(0, epochs - 1, 100)).astype(int)
        self.model.log['train/loss'] = np.array(self.model.log['train/loss'])[idx]
        self.model.log['test/loss'] = np.array(self.model.log['test/loss'])[idx]
