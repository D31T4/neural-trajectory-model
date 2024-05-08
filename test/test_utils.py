import unittest
import numpy as np
import torch

from src.utils import set_intersect, set_union, haversine
from src.ml.utils import accuracy

class set_union_tests(unittest.TestCase):
    '''
    unit test of `set_union`
    '''
    def test_empty(self):
        self.assertSetEqual(set_union(), set())

    def test_union_empty(self):
        self.assertSetEqual(set_union([], []), set())

    def test_union(self):
        self.assertSetEqual(
            set_union(['a', 'b'], ['c']),
            {'a', 'b', 'c'}
        )

class set_intersect_test(unittest.TestCase):
    '''
    unit test of `set_intersection`
    '''
    def test_empty(self):
        self.assertSetEqual(set_intersect(), set())

    def test_intersect_empty(self):
        self.assertSetEqual(set_intersect([], []), set())

    def test_intersect(self):
        self.assertSetEqual(
            set_intersect(['a', 'b', 'c'], ['b', 'c']),
            {'b', 'c'}
        )
        
class haversine_test(unittest.TestCase):
    '''
    unit test of `haversine`
    '''

    def test_empty(self):
        '''
        empty args
        '''
        out = haversine(
            np.zeros((0, 2)),
            np.zeros((0, 2))
        )

        self.assertEqual(out.shape, (0,))

    def test_one(self):
        '''
        one example from T-drive
        '''
        out = haversine(np.array([[39.88319, 116.45551]]), np.array([[39.91108, 116.58425]]))

        self.assertTrue(
            np.allclose(out, [11412.066697768278])
        )

class accuracy_test(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(
            accuracy(torch.tensor([[1.0, 1.1, 0], [1.5, 1.0, -1.0]], dtype=torch.float32), torch.tensor([1, 0])),
            1
        )

if __name__ == '__main__':
    unittest.main()