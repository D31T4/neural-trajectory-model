import unittest
import numpy as np

from src.eval.metrics import top_k, uniqueness, greedy_match, similarity, uniqueness_list
from src.data_preprocess.trajectory import Trajectory

# mock points
A = (0, 1)
B = (0, 0)
C = (1, 0)
D = (2, 0)

class top_k_test(unittest.TestCase):
    def test(self):
        trajectories = [
            Trajectory([A, B, C]),
            Trajectory([B, C, D]),
            Trajectory([A, A, A]),
            Trajectory([B])
        ]

        out = top_k(trajectories, 3)
        self.assertListEqual(out, [A, B, C])

class uniqueness_test(unittest.TestCase):
    def test(self):
        trajectories = [{
            'a': Trajectory([A, B, C]),
            'b': Trajectory([C, B, A, A]),
            'c': Trajectory([D])
        }]

        out = uniqueness(trajectories, 3)
        self.assertAlmostEqual(out, 1/3)

class uniqueness_list_test(unittest.TestCase):
    def test(self):
        trajectories = [
            Trajectory([A, B, C]),
            Trajectory([C, B, A, A]),
            Trajectory([D])
        ]

        out = uniqueness_list(trajectories, 3)
        self.assertAlmostEqual(out, 1/3)

class greedy_match_test(unittest.TestCase):
    def test_empty(self):
        t = Trajectory([A, B, C, D])

        self.assertListEqual(greedy_match([], []), [])
        self.assertListEqual(greedy_match([t], []), [-1])
        self.assertListEqual(greedy_match([], [t]), [])

    def test_match(self):
        s0 = [
            Trajectory([A, B, C, D]),
            Trajectory([B, C, D, A]),
            Trajectory([A, A, C, D]),
            Trajectory([B, C, A, A])
        ]

        s1 = [
            Trajectory([A, B, C, D]),
            Trajectory([A, A, D, C]),
            Trajectory([B, C, D, D]),
            Trajectory([B, C, A, D])
        ]

        self.assertListEqual(
            greedy_match(s0, s1),
            [0, 2, 1, 3]
        )

class similarity_test(unittest.TestCase):
    def test(self):
        s0 = np.array([
            Trajectory([A, B, C, D]),
            Trajectory([B, C, D, A]),
            Trajectory([A, A, C, D]),
            Trajectory([B, C, A, A])
        ])

        s1 = np.array([
            Trajectory([A, B, C, D]),
            Trajectory([A, A, D, C]),
            Trajectory([B, C, D, D]),
            Trajectory([B, C, A, D])
        ])

        self.assertAlmostEqual(
            similarity(s0, s1).mean(),
            (1 + 0.25 + 0.25 + .75) / 4
        )

if __name__ == '__main__':
    unittest.main()