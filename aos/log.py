import matplotlib.pyplot as plt
import collections


class Logger(collections.UserDict):
    """
    Simple logger to store key/value pairs.
    """

    def append(self, key, value):
        """
        Append a key value pair.

        Repeaded calls with same key appends.

        >>> l = Logger()
        >>> l.append('a', 1)
        >>> l.append('a', 2)
        >>> l['a']
        [1, 2]
        """
        try:
            self[key].append(value)
        except KeyError:
            self[key] = [value]
        except AttributeError:
            raise TypeError(f'Cannot append to type {type(self[key])}.')

    def get(self, key):
        return self._log[key]


class SummaryWriterMock:
    """
    Replacement for Tensorboard `SummaryWriter`.

    This class (partially) implements the same interface as the
    Tensorboard class `SummaryWriter`. It can be used as a
    replacement to directly show logged results on screen
    instead of storing to disk.

    Ben Keene: Accepts a fake log_dir to match the SummaryWriter interface.
    """

    def __init__(self, plot=True, write=False, log_dir=None):

        self.plot = plot
        self.figures = {}
        self.hparams = []

    def add_figure(self, key, fig):

        self.figures[key] = fig

    def add_hparams(self, hparam_dict, metric_dict):

        self.hparams.append((hparam_dict, metric_dict))

    def close(self):

        keys = set()
        for _, md in self.hparams:
            for k in md:
                keys.add(k)
        print(f'For results {keys} see Tensorboard.')

        if self.plot:
            plt.show()
