import torch
import aos.callbacks
from . import train


class Model:

    def __init__(self, network, optimizer, train_loss_fn, test_loss_fn, epochs):

        self.network = network
        self.optimizer = optimizer
        self.train_loss_fn = train_loss_fn
        self.test_loss_fn = test_loss_fn
        self.epochs = epochs
        self.train_loop = train.train_loop
        self.test_loop = train.test_loop

    def fit(self, train_dataloader, test_dataloader=None, callbacks=[]):

        callbacks_fn = aos.callbacks.callback_fn(callbacks)
        callbacks_fn('set_model')(self)
        callbacks_fn('set_data')(train_dataloader, test_dataloader)

        for epoch in range(self.epochs):

            callbacks_fn('on_train_epoch_start')(epoch)
            self.train_loop(
                self.network,
                train_dataloader,
                self.optimizer,
                self.train_loss_fn,
                callbacks_fn('on_train_batch_end'),
            )
            callbacks_fn('on_train_epoch_end')(epoch)

            if not test_dataloader is None:
                self.test_loop(
                    self.network,
                    test_dataloader,
                    self.test_loss_fn,
                    callbacks_fn('on_test_batch_end'),
                )
                callbacks_fn('on_test_epoch_end')(epoch)

        callbacks_fn('on_experiment_end')(self.epochs)

    def predict(self, *args, **kwargs):

        return self.network(*args, **kwargs)
