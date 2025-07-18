import unittest
import os
import aos.misc.os as cos


class TestOs(unittest.TestCase):

    def setUp(self):

        self.dir = 'open.test.dir'
        self.file = f'{self.dir}/test.file'
        if os.path.exists(self.dir):
            self.fail('Error: Test file aready exists!')

    def tearDown(self):

        os.remove(self.file)
        os.rmdir(self.dir)

    def test_open(self):

        # The builtin `open` fails because it does not
        # create directories
        with self.assertRaises(FileNotFoundError):
            f = open(self.file, 'w')

        # The wrapper opens and creates the directory
        f = cos.open(self.file, 'w')
        f.close()
        self.assertTrue(os.path.exists(self.file))

    def test_open_context_manager(self):

        with cos.open(self.file, 'w') as f:
            pass
        self.assertTrue(os.path.exists(self.file))


if __name__ == '__main__':
    unittest.main()
