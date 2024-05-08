import unittest

import pandas as pd
from datetime import datetime
from src.data_preprocess.tdrive import TDrivePreprocessConfig, get_records
from test.test_trajectory import Record

# mock points
A = (0, 10)
B = (0, 0)
C = (10, 0)
D = (20, 0)

class TestTDrivePreprocessConfig(TDrivePreprocessConfig):
    def __init__(self):
        raise

    @staticmethod
    def one_day_config():
        return TDrivePreprocessConfig(
            delta_min=30,
            start_date=datetime(2008, 2, 4),
            n_day=1,
            verbose=False,
            interp_trajectory=True,
        )

class get_records_test(unittest.TestCase):
    '''
    unit test of get_records
    '''

    def test_one_record(self):
        '''
        one point test. should extrapolate from both sides.
        '''
        config = TestTDrivePreprocessConfig.one_day_config()

        df = pd.DataFrame({
            'lat': [A[0]],
            'long': [A[1]],
            'uid': ['a'],
            't': [datetime(2008, 2, 4, 12, 1)],
        })

        out = get_records(df, config)
        
        for i in range(23):
            self.assertListEqual(out[i], [Record('a', A, votes=30)])

        self.assertListEqual(out[24], [Record('a', A, votes=1)])

    def test_two_record(self):
        '''
        2 record
        '''
        config = TestTDrivePreprocessConfig.one_day_config()

        df = pd.DataFrame({
            'lat': [A[0], B[0]],
            'long': [A[1], B[1]],
            'uid': ['a', 'a'],
            't': [datetime(2008, 2, 4, 12, 1), datetime(2008, 2, 4, 13, 15)],
        })

        out = get_records(df, config)
        
        for i in range(23):
            self.assertListEqual(out[i], [Record('a', A, votes=30)])

        self.assertListEqual(out[24], [Record('a', A, votes=1), Record('a', A, votes=29)])
        self.assertListEqual(out[25], [Record('a', A, votes=30)])
        self.assertListEqual(out[26], [Record('a', A, votes=15)])

    def test_4_record(self):
        '''
        4 record
        '''
        config = TestTDrivePreprocessConfig.one_day_config()

        df = pd.DataFrame({
            'lat': [A[0], B[0], C[0], D[0]],
            'long': [A[1], B[1], C[1], D[1]],
            'uid': ['a', 'a', 'a', 'a'],
            't': [datetime(2008, 2, 4, 12, 1), datetime(2008, 2, 4, 12, 31), datetime(2008, 2, 4, 12, 55), datetime(2008, 2, 4, 13, 15)],
        })

        out = get_records(df, config)
        
        for i in range(23):
            self.assertListEqual(out[i], [Record('a', A, votes=30)])

        self.assertListEqual(out[24], [Record('a', A, votes=1), Record('a', A, votes=29)])
        self.assertListEqual(out[25], [Record('a', A, votes=1), Record('a', B, votes=24), Record('a', C, votes=5)])
        self.assertListEqual(out[26], [Record('a', C, votes=15)])

    def test_edge_right(self):
        '''
        record > date upper bound
        '''
        config = TestTDrivePreprocessConfig.one_day_config()

        df = pd.DataFrame({
            'lat': [A[0]],
            'long': [A[1]],
            'uid': ['a'],
            't': [datetime(2008, 2, 5, 12, 1)],
        })

        out = get_records(df, config)
        
        for i in range(48):
            self.assertListEqual(out[i], [Record('a', A, votes=30)])

    def test_edge_left(self):
        '''
        one record t < date lower bound
        '''
        config = TestTDrivePreprocessConfig.one_day_config()

        df = pd.DataFrame({
            'lat': [A[0]],
            'long': [A[1]],
            'uid': ['a'],
            't': [datetime(2008, 2, 1, 12, 1)],
        })

        out = get_records(df, config)
        
        for i in range(48):
            self.assertListEqual(out[i], [])

    def test_two_record_edge_left(self):
        '''
        1 out of 2 records have t < date lower bound
        '''
        config = TestTDrivePreprocessConfig.one_day_config()

        df = pd.DataFrame({
            'lat': [A[0], B[0]],
            'long': [A[1], B[1]],
            'uid': ['a', 'a'],
            't': [datetime(2008, 2, 1, 12, 1), datetime(2008, 2, 4, 12, 1)],
        })

        out = get_records(df, config)
        
        for i in range(23):
            self.assertListEqual(out[i], [Record('a', A, votes=30)])

        self.assertListEqual(out[24], [Record('a', A, votes=1)])


if __name__ == '__main__':
    unittest.main()