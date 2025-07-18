import unittest
import aos.misc.args as cargs


class TestArgs(unittest.TestCase):

    def test_fast(self):

        parser = cargs.make_parser('fast')

        args = parser.parse_args([])
        self.assertFalse(args.fast)

        args = parser.parse_args(['-f'])
        self.assertTrue(args.fast)

    def test_plot(self):

        parser = cargs.make_parser('plot')

        args = parser.parse_args([])
        self.assertFalse(args.no_plot)

        args = parser.parse_args(['-p'])
        self.assertTrue(args.no_plot)

    def test_name(self):

        parser = cargs.make_parser(cargs.Name('module'))
        args = parser.parse_args(['test.module'])
        self.assertEqual(args.module, 'test.module')

    def test_name_or_abbrev(self):

        parser = cargs.make_parser(
            cargs.NameOrAbbrev(
                'module',
                'test.',
                {
                    ':default': 'default_module',
                    ':fast': 'fast_module',
                },
            )
        )
        args = parser.parse_args(['test.module'])
        self.assertEqual(args.module, 'test.module')

        args = parser.parse_args([':fast'])
        self.assertEqual(args.module, 'test.fast_module')


if __name__ == '__main__':
    unittest.main()
