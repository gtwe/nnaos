from collections import UserDict
import inspect


class Eval:
    """
    Function wrapper for resolution in `resolve`.

    Args:

        func: A function with one argument of name `r, _, __, ___`, etc.
              In resolve, the argument is filled in by one dictionary
              out of a nestest system of dictionaries.
              `r`: The root dictionary.
              `_`: The leave dictionary that contains `Eval`.
              `__`: One above the leave dictionary.
              etc.
    """

    def __init__(self, func):

        self.func = func


def resolve(d):
    """
    Resolve `Eval` crossreferences in nested dictionaries.

    The resolution recurses into dictionary subclasses of the same
    type ad `d`.

    Example:

    >>> param = {
    ...     'm': 2,
    ...     'n': Eval(lambda r: 2*r['m']),
    ... }
    >>> print(resolve(param))
    {'m': 2, 'n': 4}

    Example:

    Use argument names `_`, `__`, etc. to refer to intermediate dictionaries,
    similar to `.`, `..`, etc. in file systems.

    >>> param = {
    ...     'd1' : {
    ...         'd2': {
    ...             'a': 1,
    ...             'b': Eval(lambda _: _['a']),
    ...             'c': Eval(lambda __: __['d3']['d']),
    ...         },
    ...         'd3': {
    ...             'd': 2,
    ...         },
    ...     }
    ... }
    >>> print(resolve(param))
    {'d1': {'d2': {'a': 1, 'b': 1, 'c': 2}, 'd3': {'d': 2}}}


    Example:

    The ``Eval`` items can also reference back to nested and
    ``Product`` items.

    >>> import aos.misc.dicttools.product as product
    >>> params = {
    ...     'matrix': {
    ...         'm': product.Product([2,3]),
    ...         'n': 10,
    ...     },
    ...     'n_steps': Eval(lambda r: 2*r['matrix']['m']),
    ... }
    >>> params = map(resolve, product.resolve(params))
    >>> for p in params: print(p)
    {'matrix': {'m': 2, 'n': 10}, 'n_steps': 4}
    {'matrix': {'m': 3, 'n': 10}, 'n_steps': 6}

    Example:

    ``Eval`` items can depend on other ``Eval`` items.
    They are recursively resolved.

    >>> params = {
    ... 'm': Eval(lambda r: 2*r['n']),
    ... 'n': Eval(lambda r: 2),
    ... 'p': Eval(lambda r: 2*r['m'])
    ... }
    >>> print(resolve(params))
    {'m': 4, 'n': 2, 'p': 8}

    """

    return _resolve([d], recurse_type=type(d))


def _resolve(d_path, recurse_type):
    """
    Similar to `resolve` with a path of nested dictionaries.

    Args:
        d_path: A path of dictionaries to properly fill in
                variables `r, _, __`, etc. in `Eval` functions.

        recurse_type: Recurse only into subdictionaries of the
                      given type.
    """
    # print('\n', d_path)
    d_root = d_path[0]
    d_current = d_path[-1]

    needs_rerun = False
    for key, value in d_current.items():

        if isinstance(value, Eval):
            # The function `_eval` throws an _ValueIsEvalError if
            # value.func tries to access a key with an un-resolved
            # Eval . In this case, we resolve all other Evals that
            # we can and then re-run.
            try:
                d_current[key] = _eval(value.func, d_path)
            except _ValueIsEvalError:
                needs_rerun = True

        if type(value) is recurse_type:
            _resolve(d_path+[value], recurse_type)

    if needs_rerun:
        _resolve([d_root], recurse_type)

    return d_current


def _eval(func, d_path):
    """
    Evaluate `func` with one dictionary in the list `d_path`, dependent or argument name.

    The evaluation throws an _ValueIsEvalError if it tries to use a
    unevaluated `Eval` object.

    Args:

        func: A function with one argument of name `r, _, __, ___`, etc.

    Returns:

        `func` evaluated on the dictionary refered to by the argument name,
        see `Eval` for details.
    """
    signature = inspect.signature(func)
    assert len(
        signature.parameters) == 1, "Error: Eval resolution requires functions in one argument."
    arg_name = next(iter(signature.parameters.keys()))

    # In all function evaluations, we wrap the dictionary
    # in _EvalDict, to throw an _ValueIsEvalError if
    # the function evaluation refers to an unevaluated `Eval.`

    if arg_name == len(arg_name)*'_':
        up = len(arg_name)
        return func(_eval_dict(d_path[-up]))
    elif arg_name == 'r':
        d_root = d_path[0]
        return func(_eval_dict(d_root))
    else:
        raise NameError('Function argument names must be in r, _, __, etc.')


class _ValueIsEvalError(Exception):
    pass


def _eval_dict(d):

    class _EvalDict(type(d)):

        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)

        def __getitem__(self, key):

            value = super().__getitem__(key)

            if isinstance(value, Eval):
                raise _ValueIsEvalError
            else:
                return value

    return _EvalDict(d)
