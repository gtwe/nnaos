import doctest
import unittest

import aos.log
import aos.misc.dicttools

modules = [
    aos.misc.dicttools,
    aos.misc.dicttools.cdict,
    aos.misc.dicttools.eval,
    aos.misc.dicttools.product,
    aos.log,
]


# For automatic unittest test discovery
def load_tests(loader, tests, ignore):
    for m in modules:
        tests.addTests(doctest.DocTestSuite(m))
    return tests


if __name__ == '__main__':
    # Run with command line option `-v` to see a verbose output.
    # Careful, there is no joint summary.
    for m in modules:
        doctest.testmod(m)
