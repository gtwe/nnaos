import copy
import itertools


class Product:
    """
    List and group, to be resolved with `resolve`.

    Parameters:
        items: List
        group: Identifier such as int or string

    """

    def __init__(self, items: list, group=None):

        self.items = items
        self.group = group


def _find_product(d):
    """
    Find instances if `Product` in nested dictionaries `d`.

    Args:
        d (dict): (nested) dictionary.

    Returns:
        Genertor of all `Product` instances.
    """
    for value in d.values():

        if type(value) is dict:
            for p in _find_product(value):
                yield p
        elif isinstance(value, Product):
            yield value


def _bubble_up(d, group):
    """
    same as `resolve` for a single group and non-nested dictionary.
    """
    # Build a list of dictionaries with the key,value pairs for each
    # item of the ``Product`` in the given group.
    #
    # For the line ((k, v) for ...) we use a generator object instead
    # of a list. This allows infinite length `Product`s, e.g.
    # `Product(itertools.count(), group=0)` to generate indices,
    # see `product_idx` for an example.
    #
    # Although we always have k=key, the itertools.repeat(key) is
    # necessary, because otheriwise the lazy execution of the generator
    # will use the last key instead of the actual key.

    d_overwrites = [
        # [(key, v) for v in value.items]
        ((k, v) for k, v in zip(itertools.repeat(key), value.items))
        for key, value in d.items() if isinstance(value, Product) and value.group == group
    ]
    d_overwrites = map(dict, zip(*d_overwrites))
    ds = [{**copy.deepcopy(d), **do} for do in d_overwrites]

    # Join the new key,value pairs into the full dictionaries
    if len(ds) == 0:
        # return Product([d], group)
        return d
    else:
        return Product(ds, group)


def _resolve_group(d, group):
    """
    Recursively apply `bubble_up` to nested dictionaries.

    Same as `resolve_group` but only for a single dictionary.
    If the input does not contain `Product` with `group`,
    return the dictionary itself.
    """
    for key, value in d.items():
        if type(value) is dict:
            d[key] = _resolve_group(value, group)

    return _bubble_up(d, group)

def resolve_group(ds, group):
    """"
    Same as `resolve` restricted to the given group.
    """
    if type(ds) is dict:
        ds = [ds]

    resolved = (_resolve_group(d, group) for d in ds)
    resolved = (r.items if isinstance(r, Product) else [r] for r in resolved)
    return list(itertools.chain(*resolved))


def resolve(ds):
    """
    Make a list of dictionaries from a (nested) dictionary of lists.

    Creates a list of nested dictionaries, where each `Product`
    value is replaced with one item from the list inside `Product`.

    Multiple `Product`s of the same group generate one output
    dictionary per product list item. Multiple `Product`s of
    different groups are treated as an outer product.

    Args:
        ds (dict or [dict]): Dictionary or list of dictionaries

    Returns:
        A list of dictionaries with resolved `Product`s.

    Example: No or same group matches entries between different products

    >>> params = {
    ...     'm': Product([1, 2]),
    ...     'n': Product([3, 4]),
    ...     'p': '5'
    ... }
    >>> params = resolve(params)
    >>> for p in params: print(p)
    {'m': 1, 'n': 3, 'p': '5'}
    {'m': 2, 'n': 4, 'p': '5'}

    Example: Different groups produce all combinations

    >>> params = {
    ...     'm': Product([1, 2], group=1),
    ...     'n': Product([3, 4], group=2),
    ...     'p': '5'
    ... }
    >>> params = resolve(params)
    >>> for p in params: print(p)
    {'m': 1, 'n': 3, 'p': '5'}
    {'m': 1, 'n': 4, 'p': '5'}
    {'m': 2, 'n': 3, 'p': '5'}
    {'m': 2, 'n': 4, 'p': '5'}

    Example: Nested dictionaries with multiple parameters

    >>> params = {
    ...     'subdict': {
    ...         'subsubdict': {
    ...             'a': Product([1, 2], group='c'),
    ...         },
    ...         'b': Product([11, 12], group='b'),
    ...     },
    ...     'c': Product([21, 22], group='c'),
    ... }
    >>> params = resolve(params)
    >>> for p in params: print(p)
    {'subdict': {'subsubdict': {'a': 1}, 'b': 11}, 'c': 21}
    {'subdict': {'subsubdict': {'a': 1}, 'b': 12}, 'c': 21}
    {'subdict': {'subsubdict': {'a': 2}, 'b': 11}, 'c': 22}
    {'subdict': {'subsubdict': {'a': 2}, 'b': 12}, 'c': 22}

    Example:

    For multiple ``Product``s, the resulting list contains
    all combinations. If that is not intended, one can combine
    several products by the ``group`` argument.

    >>> params = {
    ...     'matrix': {
    ...         'm': 2,
    ...         'n': Product([3,4], group=1),
    ...     },
    ...     'lr': Product([0.1, 0.2]),
    ...     'n_steps': Product([10, 20], group=1),
    ... }
    >>> params = resolve(params)
    >>> for p in params: print(p)
    {'matrix': {'m': 2, 'n': 3}, 'lr': 0.1, 'n_steps': 10}
    {'matrix': {'m': 2, 'n': 3}, 'lr': 0.2, 'n_steps': 10}
    {'matrix': {'m': 2, 'n': 4}, 'lr': 0.1, 'n_steps': 20}
    {'matrix': {'m': 2, 'n': 4}, 'lr': 0.2, 'n_steps': 20}

    Example:

    Products can be nested (Product inside Product)
    For correct resolution, we force resolution of the outer
    Product first.

    >>> params = {
    ...     'experiment': Product(['ex1', 'ex2'], group='experiment'),
    ...     'network': Product([
    ...         {
    ...             'width': 1,
    ...             'depth': Product([1,2]),
    ...         },
    ...         {
    ...             'n_weights': 2,
    ...         },
    ...     ], group='experiment'),
    ...     'lr': 0.1,
    ... }
    >>> params = resolve_group(params, 'experiment')
    >>> params = resolve(params)
    >>> for p in params: print(p)
    {'experiment': 'ex1', 'network': {'width': 1, 'depth': 1}, 'lr': 0.1}
    {'experiment': 'ex1', 'network': {'width': 1, 'depth': 2}, 'lr': 0.1}
    {'experiment': 'ex2', 'network': {'n_weights': 2}, 'lr': 0.1}

    """
    if type(ds) is dict:
        ds = [ds]

    try:

        group = next(iter(itertools.chain(*(_find_product(d)
                                            for d in ds)))).group
        ds = resolve_group(ds, group)

        return resolve(ds)

    except StopIteration:
        return ds


def product_idx(group=None):
    """
    Add an index for dictionary `Product`s.

    Example:

    >>> params = {
    ...     'a': {
    ...         'b': Product([10, 11, 12], group=0),
    ...     },
    ...     'idx': product_idx(group=0)
    ... }
    >>> params = resolve(params)
    >>> for p in params: print(p)
    {'a': {'b': 10}, 'idx': 0}
    {'a': {'b': 11}, 'idx': 1}
    {'a': {'b': 12}, 'idx': 2}
    """
    return Product(itertools.count(), group)


def get_keys(d, keys=None, path=''):
    """
    For all ``Product`` items, extract a string for the (nested) key.

    Example:

    >>> params = {
    ...         'matrix': {
    ...         'm': 2,
    ...         'n': Product([3,4], group=1),
    ...     },
    ...     'lr': Product([0.1, 0.2]),
    ...     'n_steps': Product([10, 20], group=1),
    ...     }
    >>> print(get_keys(params))
    ['matrix.n', 'lr', 'n_steps']

    """
    # Do not use default argument ``keys=[]`` and change the list.
    keys = [] if keys is None else keys

    for key, value in d.items():
        if isinstance(value, Product):
            keys.append(path+key)
        elif type(value) is dict:
            keys += get_keys(value, keys=[], path=path+key+'.')

    return keys
