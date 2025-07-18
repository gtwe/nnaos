import unittest
import aos.misc.dicttools as dicttools


class TestProduct(unittest.TestCase):

    def test_copy_subdicts(self):

        d = {
            'subdict': {
                'a': 0
            },
            'b': dicttools.Product([0, 1])
        }
        ds = dicttools.product.resolve(d)

        # Each product element should contain an independent
        # copy of the 'subdict'.
        #
        # To verifty, we change the value of 'a' for
        # b=0 and verifty that the value of 'a' for
        # b=1 remains unchanged.

        old_value_a_for_b0 = ds[0]['subdict']['a']
        ds[0]['subdict']['a'] = 7
        new_value_a_for_b0 = ds[0]['subdict']['a']
        value_a_for_b1 = ds[1]['subdict']['a']

        self.assertEqual(value_a_for_b1, old_value_a_for_b0)
        self.assertNotEqual(value_a_for_b1, new_value_a_for_b0)


if __name__ == '__main__':
    unittest.main()
