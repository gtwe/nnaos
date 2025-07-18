import argparse


class Name:

    def __init__(self, name):

        self.name = name


class NameOrAbbrev:

    def __init__(self, name, base, abbrevs):

        self.name = name
        self.base = base
        self.abbrevs = abbrevs

    def expand(self, s):

        if s in self.abbrevs:
            return self.base + self.abbrevs[s]
        else:
            return s


def make_parser(*args):

    parser = argparse.ArgumentParser()

    for arg in args:

        if arg == 'fast':
            parser.add_argument(
                '-f', '--fast', help='Execute a fast test version of the script', action='store_true')

        elif arg == 'plot':
            parser.add_argument('-p', '--no-plot',
                                help='Suppress plots', action='store_true')

        elif arg == 'tensorboard':
            parser.add_argument('-t', '--no-tensorboard',
                                help='No Tensorboard logging', action='store_true')


        elif isinstance(arg, Name):
            parser.add_argument(arg.name, help=f'Provide {arg.name}')

        elif isinstance(arg, NameOrAbbrev):
            parser.add_argument(arg.name, help=f'Provide {arg.name} or abbreviation in {list(arg.abbrevs)}', type=arg.expand)

    return parser
