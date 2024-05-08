import unittest
import numpy as np
from sklearn.metrics import euclidean_distances

from src.data_preprocess.trajectory import Trajectory
from src.main.baseline import DailyTrajectoryRecovery, get_aggregated_observations
from test.test_trajectory import A, B, C, D

class get_aggregated_observations_test(unittest.TestCase):
    '''
    unit test of `get_aggregated_observations`
    '''
    def test(self):
        trajectories = [
            Trajectory([A, B, C, D]),
            Trajectory([A, C, C, D]),
            Trajectory([C, C, D, A])
        ]

        self.assertDictEqual(
            dict(get_aggregated_observations(trajectories, 0)),
            { A: 2, C: 1 }
        )

        self.assertDictEqual(
            dict(get_aggregated_observations(trajectories, 1)),
            { B: 1, C: 2 }
        )

        self.assertDictEqual(
            dict(get_aggregated_observations(trajectories, 2)),
            { C: 2, D: 1 }
        )

        self.assertDictEqual(
            dict(get_aggregated_observations(trajectories, 3)),
            { D: 2, A: 1 }
        )

def euclidean_matrix(trajectories: np.ndarray, candidates: np.ndarray, tstamp: int):
    next_loc = trajectories[:, -1]
    return euclidean_distances(next_loc, candidates)

class DailyTrajectoryRecoveryTest(unittest.TestCase):
    def test_unaggregate_observations(self):
        '''
        test `unaggregate_observations`
        '''
        aggregated_observation = [
            (A, 1),
            (B, 2),
            (D, 3),
            (C, 4)
        ]

        unaggregated_observations, indices = DailyTrajectoryRecovery.unaggregate_observations(aggregated_observation)

        self.assertTrue(np.all(
            unaggregated_observations == np.array([
                A, B, B, D, D, D, C, C, C, C
            ])
        ))

        self.assertTrue(np.all(
            indices == np.array([
                0, 1, 1, 2, 2, 2, 3, 3, 3, 3
            ])            
        ))

        # duplicated record test
        aggregated_observation = [
            (A, 1),
            (B, 2),
            (A, 3)
        ]

        unaggregated_observations, indices = DailyTrajectoryRecovery.unaggregate_observations(aggregated_observation)

        self.assertTrue(np.all(
            unaggregated_observations == np.array([
                A, B, B, A, A, A
            ])
        ))

        self.assertTrue(np.all(
            indices == np.array([
                0, 1, 1, 2, 2, 2
            ])
        ))

    def test_consistency(self):
        '''
        unit test of `run`.

        check prediction is consistent with aggregated observations
        '''
        ground_truth = [
            Trajectory([A, B, C, D]),
            Trajectory([A, A, C, C]),
            Trajectory([A, C, D, A]),
            Trajectory([B, A, D, D])
        ]

        dailyRecovery = DailyTrajectoryRecovery(
            get_observations=lambda tstamp: get_aggregated_observations(ground_truth, tstamp),
            cost_matrix=euclidean_matrix,
            sequence_length=4,
            sequence_count=4
        )

        dailyRecovery.run(verbose=False)
        predicted = dailyRecovery.get_predicted_trajectories()

        for i in range(4):
            self.assertDictEqual(
                dict(get_aggregated_observations(predicted, i)),
                dict(get_aggregated_observations(ground_truth, i))
            )

    def test_get_predicted_trajectories(self):
        '''
        unit test of `get_predicted_trajectories`
        '''
        ground_truth = [
            Trajectory([A, B, C, D]),
            Trajectory([A, A, C, C]),
            Trajectory([A, C, D, A]),
            Trajectory([B, A, D, D])
        ]

        dailyRecovery = DailyTrajectoryRecovery(
            get_observations=None,
            cost_matrix=None,
            sequence_length=4,
            sequence_count=4
        )

        dailyRecovery.result = np.array(ground_truth)

        predicted_trajectories = dailyRecovery.get_predicted_trajectories()

        self.assertListEqual(ground_truth, predicted_trajectories)

    def test_run(self):
        '''
        test `run`.

        check recovery with trivial task
        '''
        ground_truth = [
            Trajectory([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]),
            Trajectory([(0.0, 10.0), (0.0, 11.0), (0.0, 12.0), (0.0, 13.0)])
        ]

        dailyRecovery = DailyTrajectoryRecovery(
            get_observations=lambda tstamp: get_aggregated_observations(ground_truth, tstamp),
            cost_matrix=euclidean_matrix,
            sequence_length=4,
            sequence_count=2
        )

        dailyRecovery.run(verbose=False)
        
        predicted = dailyRecovery.get_predicted_trajectories()

        for t1, t2 in zip(predicted, ground_truth):
            self.assertEqual(t1, t2)

    def test_one(self):
        ground_truth = [Trajectory([A, B, A, A, D])]

        dailyRecovery = DailyTrajectoryRecovery(
            get_observations=lambda tstamp: get_aggregated_observations(ground_truth, tstamp),
            cost_matrix=euclidean_matrix,
            sequence_length=5,
            sequence_count=1
        )

        dailyRecovery.run(verbose=False)

        predicted = dailyRecovery.get_predicted_trajectories()
        self.assertListEqual(predicted, ground_truth)