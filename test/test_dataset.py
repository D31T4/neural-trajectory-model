import unittest
import torch

from src.ml.dataset import hour_sequence, trajectory_to_tensor, create_point_to_class_map
from src.data_preprocess.trajectory import Trajectory

# mock points
A = (0, 1)
B = (0, 0)
C = (1, 0)
D = (2, 0)

class hour_sequence_test(unittest.TestCase):
    '''
    unit test of `hour_sequence`
    '''
    def test(self):
        self.assertListEqual(
            hour_sequence(48),
            [i // 2 for i in range(48)]
        )

        self.assertListEqual(
            hour_sequence(12),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        )

        self.assertListEqual(
            hour_sequence(8),
            [0, 3, 6, 9, 12, 15, 18, 21]
        )

class trajectory_to_tensor_test(unittest.TestCase):
    '''
    unit test of `trajectory_to_tensor`
    '''
    def test(self):
        t = Trajectory([A, B, C, D])

        expected = torch.tensor([A, B, C, D])

        self.assertTrue(torch.all(torch.eq(
            trajectory_to_tensor(t),
            expected
        )))

class create_point_to_class_map_test(unittest.TestCase):
    def test(self):
        points = [A, B, C, D]
        map = create_point_to_class_map(torch.tensor(points))

        self.assertEqual(map[A], 0)
        self.assertEqual(map[B], 1)
        self.assertEqual(map[C], 2)
        self.assertEqual(map[D], 3)

if __name__ == '__main__':
    unittest.main()