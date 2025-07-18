from collections import abc


# Alternative Implementation:
#
# Inherit form abc.Mapping to ensure that this class is a
# Mapping and thus can be used as kewyword arguments
# in function calls.
#
class CDict(dict):
    """
    Dictionary with class style element access.

    Instance behave as a regular dictionary, with the difference that its
    keys can be accessed like a class attribute.

    Careful:

        The mechanism only works if the name is not already contained in the
        class like e.g. `clear`, `copy`, `get`, `items`, `pop`, `update`,
        `values`, etc.

    Examples:

    >>> d = CDict({'a': 1, 'b': 2})
    >>> print(d.a)
    1

    Modify existing elements
    >>> d['a'] = 3
    >>> print(d.a)
    3
    >>> d.a = 4
    >>> print(d['a'])
    4

    Add new elements
    >>> d['c'] = 4
    >>> print(d.c)
    4
    >>> d.d = 5
    >>> print(d['d'])
    5

    The class can be used in keyword arguments as usual:
    >>> def f(a, b, c, d):
    ...     return None
    >>> f(**d)

    Update multiple elements
    >>> d.update({'a': 6, 'e': 7})
    >>> print(d)
    {'a': 6, 'b': 2, 'c': 4, 'd': 5, 'e': 7}

    Iteration as for dictionaries.
    >>> d = CDict({'a': 2})
    >>> for k, v in d.items():
    ...     print(k, v)
    a 2

    Class members cannot be overridden.
    >>> d.update = 5
    Traceback (most recent call last):
     ...
    AttributeError: Cannot override CDict attribute update.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError()

    def __setattr__(self, name, value):

        # UserDict stores the dictionary internally as `data`.

        if hasattr(self, name) and not name in self:
            raise AttributeError(f'Cannot override CDict attribute {name}.')
        else:
            self[name] = value


def cn_dict(d: dict) -> CDict:
    """
    Transform nested `dict`s into `CDict`s.

    Example:

    >>> d = {
    ...     'a': {
    ...         'b': {
    ...             'c': 1,
    ...             'd': 2,
    ...         }
    ...     }
    ... }
    >>> d = cn_dict(d)
    >>> print(d.a.b.c)
    1
    """
    for key, value in d.items():
        if type(value) is dict:
            d[key] = cn_dict(value)

    return CDict(d)


#
# Alternative Implementation of CDict.
#
# Advantage: No conflict between dictinary keys and
#            dictionary methods like e.g.
#            d.update versus d['update'] in CDict
#
# Disadvantage: Less flexible, i.e. has no `update` method.
#
# Inheriting form abc.Mapping ensures that this class is a
# Mapping and thus can be used as kewyword arguments
# in function calls.
class ClassFromDict(abc.Mapping):
    """





    Invalidate docstring for deprecated class to
    skip doctests ...





    """

    """
    Recursively transforms a dictionary to a class.

    Example 1:

    >>> ex_dict = {'a': 1, 'b': 2}
    >>> ex_class = ClassFromDict(ex_dict)
    >>> ex_class.a
    1
    >>> ex_class.b
    2

    Example 2 (recursive):

    >>> ex_dict = {'a': 1, 'bc': {'b': 2, 'c':3}}
    >>> ex_class = ClassFromDict(ex_dict)
    >>> ex_class.a
    1
    >>> ex_class.bc.b
    2
    >>> ex_class.bc.c
    3

    The generated classes can be used as keyword arguments
    in function calls.

    >>> def f(a, b):
    ...     return a*b
    >>> ex_dict = {'a': 2, 'b': 3}
    >>> ex_class = ClassFromDict(ex_dict)
    >>> f(**ex_dict)
    6
    >>> f(**ex_class)
    6

    The class can also be used like a dictionary, e.g.
    we can iterate over key, value pairs

    >>> ex_dict = {'a': 2}
    >>> ex_class = ClassFromDict(ex_dict)
    >>> for k, v in ex_class.items():
    ...     print(k, v)
    a 2

    """

    def __init__(self, d: dict):

        raise DeprecationWarning()

        for key, value in d.items():
            if type(value) is dict:
                setattr(self, key, ClassFromDict(value))
            else:
                setattr(self, key, value)

    def dict(self):

        return self.__dict__

    def __getitem__(self, key):

        return self.__dict__[key]

    def __iter__(self):

        return self.__dict__.__iter__()

    def __len__(self):

        return self.__dict__.__len__()
