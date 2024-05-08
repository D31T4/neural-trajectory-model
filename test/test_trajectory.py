'''
unit test of __init__.py
'''

import unittest

import numpy as np
from datetime import datetime
from src.data_preprocess.trajectory import PreprocessConfig, Trajectory, TrajectoryBuilder, to_dataframe, from_dataframe, Record, aggregate_records, get_trajectories_one_day

# mock points
A = (0., 1.)
B = (0., 0.)
C = (1., 0.)
D = (2., 0.)

class TestPreprocessConfig(PreprocessConfig):
    def __init__(self):
        raise

    @staticmethod
    def one_day_config():
        return PreprocessConfig(
            delta_min=30,
            start_date=datetime(2014, 6, 2),
            n_day=1,
            verbose=False
        )

class TrajectoryTest(unittest.TestCase):
    '''
    unit test of `Trajectory`
    '''
    def test_visit_count(self):
        '''
        test `visit_count` method
        '''
        t = Trajectory([A, B, C, D])
        self.assertDictEqual(t.visit_count(), { A: 1, B: 1, C: 1, D: 1 })

        t = Trajectory([A, A, C, C])
        self.assertDictEqual(t.visit_count(), { A: 2, C: 2 })

    def test_eq(self):
        '''
        test `Trajectory.__eq__`
        '''
        self.assertEqual(
            Trajectory([A, B, C, D]),
            Trajectory([A, B, C, D])
        )

        self.assertNotEqual(
            Trajectory([A, B, C, D]),
            Trajectory([A, B, C])
        )

        self.assertNotEqual(
            Trajectory([A, B, C, D]),
            Trajectory([B, C, D, A])
        )

    def test_similarity(self):
        '''
        test `Trajectory.similarity`
        '''
        self.assertAlmostEqual(
            Trajectory.similarity(
                Trajectory([A, B, C, D]),
                Trajectory([A, B, C, D])
            ),
            1
        )

        self.assertAlmostEqual(
            Trajectory.similarity(
                Trajectory([B, B, C, D]),
                Trajectory([A, B, C, D])
            ),
            .75
        )

        self.assertAlmostEqual(
            Trajectory.similarity(
                Trajectory([A, B, C, D]),
                Trajectory([B, C, D, A])
            ),
            0
        )

    def test_k_most_frequently_visited(self):
        '''
        test of `k_most_frequently_visited`
        '''
        t = Trajectory([A, B, C, D])
        self.assertListEqual(t.k_most_frequently_visited(3), [(A, 1), (B, 1), (C, 1)])

        t = Trajectory([A, A, C, C])
        self.assertListEqual(t.k_most_frequently_visited(3), [(A, 2), (C, 2)])

        t = Trajectory([A, A, A, B, B, C, C, D])
        self.assertListEqual(t.k_most_frequently_visited(3), [(A, 3), (B, 2), (C, 2)])

class TrajectoryBuilderTest(unittest.TestCase):
    '''
    unit test of `TrajectoryBuilder`
    '''
    
    def test_init(self):
        '''
        test `TrajectoryBuilder.__init__`
        '''
        self.assertTrue(
            TrajectoryBuilder([A, B, C]).build() == Trajectory([A, B, C])
        )

    def test_interpolate(self):
        '''
        test `TrajectoryBuilder.test_interpolate`
        '''

        # front extrapolate: _->_->_->A => A->A->A->A
        self.assertTrue(
            TrajectoryBuilder([None] * 3 + [A]).interpolate(TrajectoryBuilder.interp_nearest_fill).build() == Trajectory([A] * 4)
        )

        # end extrapolate: A->_->_ => A->A->A
        self.assertTrue(
            TrajectoryBuilder([A] + [None] * 2).interpolate(TrajectoryBuilder.interp_nearest_fill).build() == Trajectory([A] * 3)
        )

        # interpolate: A->_->_->_->B => A->A->A->B->B
        self.assertTrue(
            TrajectoryBuilder([A] + [None] * 3 + [B]).interpolate(TrajectoryBuilder.interp_nearest_fill).build() == Trajectory([A] * 3 + [B] * 2)
        )

        # interpolate: A->_->_->_->B->B->_->_->_->_->C => A->A->A->B->B->B->B->B->C->C->C
        self.assertTrue(
            TrajectoryBuilder([A] + [None] * 3 + [B] * 2 + [None] * 4 + [C]).interpolate(TrajectoryBuilder.interp_nearest_fill).build() == Trajectory([A] * 3 + [B] * 5 + [C] * 3)
        )

        # mix: _->A->_->_->_->B->_->_ => A->A->A->A->B->B->B->B
        self.assertTrue(
            TrajectoryBuilder([None] + [A] + [None] * 3 + [B] + [None] * 2).interpolate(TrajectoryBuilder.interp_nearest_fill).build() == Trajectory([A] * 4 + [B] * 4)
        )

        # extrapolate: _->_->A->_ => A->A->A->A
        self.assertTrue(
            TrajectoryBuilder([None] * 2 + [A] + [None]).interpolate(TrajectoryBuilder.interp_nearest_fill).build() == Trajectory([A] * 4)
        )

class aggregate_records_test(unittest.TestCase):
    '''
    unit test of `aggregate_records`
    '''
    
    def test_empty(self):
        '''
        empty list
        '''
        records = [[]]

        records = aggregate_records(records, TestPreprocessConfig.one_day_config())

        self.assertListEqual(records[0], [])

    def test_one(self):
        '''
        one user
        '''
        records = [[Record('a', A, 1)]]

        records = aggregate_records(records, TestPreprocessConfig.one_day_config())

        self.assertListEqual(records[0], [Record('a', A, 1)])

    def test_many_unique(self):
        '''
        one user, multiple records, unique locations
        '''
        records = [[Record('a', A, 1), Record('a', B, 3), Record('a', C, 10)]]

        records = aggregate_records(records, TestPreprocessConfig.one_day_config())

        self.assertListEqual(records[0], [Record('a', C, 10)])

    def test_many_duplicate(self):
        '''
        one user, multiple records, duplicate locations
        '''
        records = [[Record('a', A, 4), Record('a', A, 3), Record('a', C, 5)]]

        records = aggregate_records(records, TestPreprocessConfig.one_day_config())

        self.assertListEqual(records[0], [Record('a', A, 7)])

    def test_mixed(self):
        '''
        multiple users, mixed scenario
        '''
        records = [[
            Record('a', A, 4), Record('a', B, 3), Record('a', C, 10),
            Record('b', B, 1), Record('b', B, 5),
            Record('c', C, 5), Record('c', C, 6), Record('c', A, 1)
        ]]

        records = aggregate_records(records, TestPreprocessConfig.one_day_config())

        self.assertListEqual(
            records[0],
            [Record('a', C, 10), Record('b', B, 6), Record('c', C, 11)]
        )

class get_trajectories_one_day_test(unittest.TestCase):
    def test_empty(self):
        config = TestPreprocessConfig.one_day_config()
        config.n_window_per_day = 0

        self.assertDictEqual(
            get_trajectories_one_day([], config),
            dict()
        )

class to_dataframe_test(unittest.TestCase):
    '''
    unit test of `to_dataframe`
    '''
    def test(self):
        t = { 'a': Trajectory([A, B, C, D]) }
        df = to_dataframe(t, 4)

class from_dataframe_test(unittest.TestCase):
    '''
    unit test of `from_dataframe`
    '''
    def test(self):
        t0 = { 'a': Trajectory([A, B, C, D]) }
        t1 = from_dataframe(to_dataframe(t0, 4), 4)

        self.assertEqual(t0, t1)

if __name__ == '__main__':
    unittest.main()
