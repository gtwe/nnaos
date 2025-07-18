from . import cdict
from . import eval
from . import product

from .product import Product, product_idx
from .cdict import CDict, cn_dict
from .eval import Eval

def resolve(d: dict) -> CDict:
    """
    Resolve `Product`, `Eval` and transer to class.

    Example:

    >>> params = {
    ...     'network': {
    ...         'width': Product([5, 10], group='net'),
    ...         'depth': Eval(lambda _: _.width),
    ...         'id': product_idx(group='net'),
    ...     },
    ...     'data': {
    ...         'n_samples': Eval(lambda r: r.network.width * r['network']['depth']),
    ...     },
    ... }
    >>> params = resolve(params)
    >>> print(params)
    [{'network': {'width': 5, 'depth': 5, 'id': 0}, 'data': {'n_samples': 25}}, {'network': {'width': 10, 'depth': 10, 'id': 1}, 'data': {'n_samples': 100}}]
    """
    ds = product.resolve(d)
    ds = map(cn_dict, ds)
    ds = map(eval.resolve, ds)
    return list(ds)
