import pathlib
import builtins


def open(file, *args, **kwargs):
    """
    Wrapper around the builtin `open` that creates directories if necessary.
    """
    f = pathlib.Path(file)
    f.parent.mkdir(exist_ok=True, parents=True)
    return builtins.open(file, *args, **kwargs)
