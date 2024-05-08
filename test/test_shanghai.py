import unittest
import pandas as pd
from datetime import datetime

from src.data_preprocess.shanghai import get_records, Record
from test.test_trajectory import TestPreprocessConfig

# mock points
A = (0, 1)
B = (0, 0)
C = (1, 0)
D = (2, 0)

class get_records_test(unittest.TestCase):
    def test_atomic(self):
        '''
        record lies within 1 window
        '''
        df = pd.DataFrame({
            'latitude': [A[0]],
            'longitude': [A[1]],
            'user id': ['a'],
            'start time': [datetime(2014, 6, 2, 0, 1)],
            'end time': [datetime(2014, 6, 2, 0, 2)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())
        
        self.assertListEqual(
            records[0],
            [Record('a', A, 1)]
        )

    def test_spanning_2(self):
        '''
        record lies across 2 windows
        '''
        df = pd.DataFrame({
            'latitude': [A[0]],
            'longitude': [A[1]],
            'user id': ['a'],
            'start time': [datetime(2014, 6, 2, 0, 1)],
            'end time': [datetime(2014, 6, 2, 0, 30)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())
        
        self.assertListEqual(
            records[0],
            [Record('a', A, 29)]
        )

        self.assertListEqual(
            records[1],
            [Record('a', A, 0)]
        )

    def test_spanning_n(self):
        '''
        record lies across n windows
        '''
        df = pd.DataFrame({
            'latitude': [A[0]],
            'longitude': [A[1]],
            'user id': ['a'],
            'start time': [datetime(2014, 6, 2, 0, 1)],
            'end time': [datetime(2014, 6, 2, 1, 30)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())
        
        self.assertListEqual(
            records[0],
            [Record('a', A, 29)]
        )

        self.assertListEqual(
            records[1],
            [Record('a', A, 30)]
        )

        self.assertListEqual(
            records[2],
            [Record('a', A, 30)]
        )

        self.assertEqual(
            records[3],
            [Record('a', A, 0)]
        )

    def test_edge_left(self):
        '''
        start date < lower bound
        lower bound < end date < upper bound
        '''
        df = pd.DataFrame({
            'latitude': [A[0]],
            'longitude': [A[1]],
            'user id': ['a'],
            'start time': [datetime(2014, 6, 1, 23, 55)],
            'end time': [datetime(2014, 6, 2, 0, 5)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())
        
        self.assertListEqual(
            records[0],
            [Record('a', A, 5)]
        )

    def test_edge_right(self):
        '''
        lower bound < start date < upper bound
        end date > upper bound
        '''
        df = pd.DataFrame({
            'latitude': [A[0]],
            'longitude': [A[1]],
            'user id': ['a'],
            'start time': [datetime(2014, 6, 2, 23, 55)],
            'end time': [datetime(2014, 6, 3, 0, 5)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())
        
        self.assertListEqual(
            records[-1],
            [Record('a', A, 5)]
        )

    def test_out_of_bound_left(self):
        '''
        end date < lower bound
        '''
        df = pd.DataFrame({
            'latitude': [A[0]],
            'longitude': [A[1]],
            'user id': ['a'],
            'start time': [datetime(2014, 6, 2, 23, 55)],
            'end time': [datetime(2014, 6, 2, 23, 58)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())
        
        self.assertEqual(
            len(records[0]),
            0
        )

    def test_out_of_bound_right(self):
        '''
        start date > upper bound
        '''
        df = pd.DataFrame({
            'latitude': [A[0]],
            'longitude': [A[1]],
            'user id': ['a'],
            'start time': [datetime(2014, 6, 3, 23, 55)],
            'end time': [datetime(2014, 6, 3, 23, 58)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())
        
        self.assertEqual(
            len(records[-1]),
            0
        )

    def test_mixed(self):
        '''
        mixed scenarios
        '''
        df = pd.DataFrame({
            'latitude': [A[0], B[0], C[0]],
            'longitude': [A[1], B[1], C[1]],
            'user id': ['a', 'b', 'c'],
            'start time': [datetime(2014, 6, 2, 23, 55), datetime(2014, 6, 1, 23, 55), datetime(2014, 6, 2, 0, 30)],
            'end time': [datetime(2014, 6, 3, 0, 5), datetime(2014, 6, 2, 0, 50), datetime(2014, 6, 2, 0, 40)]
        })

        records = get_records(df, TestPreprocessConfig.one_day_config())

        self.assertListEqual(
            records[0],
            [Record('b', B, 30)]
        )

        self.assertListEqual(
            records[1],
            [Record('b', B, 20), Record('c', C, 10)]
        )

        self.assertListEqual(
            records[-1],
            [Record('a', A, 5)]    
        )


if __name__ == '__main__':
    unittest.main()