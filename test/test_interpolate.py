import unittest

import numpy as np
import math
from datetime import datetime, timedelta
import time

from src.data_preprocess.interpolate import SpeedAwareLinearInterpolator

def date_to_float(d: datetime) -> float:
    return time.mktime(d.timetuple())

class SpeedAwareLinearInterpolatorTest(unittest.TestCase):
    def test_no_dynamic(self):
        '''
        travel time < sample time.
        '''
        vel = 1 / 60

        interpolator = SpeedAwareLinearInterpolator(
            vel, 
            30,
        )

        out_t, out_points = interpolator.interpolate(
            datetime(2014, 6, 2, 0, 0),
            datetime(2014, 6, 2, 0, 30),
            (0, 0),
            (1, 1)
        )

        self.assertListEqual(
            out_t,
            [datetime(2014, 6, 2, 0, 30) - timedelta(minutes=math.sqrt(2))]
        )

        self.assertTrue(np.allclose(
            out_points,
            np.array([
                [0, 0]
            ])
        ))

    def test_no_points(self):
        vel = 1e-10

        interpolator = SpeedAwareLinearInterpolator(
            vel, 
            30,
        )

        out_t, out_points = interpolator.interpolate(
            datetime(2014, 6, 2, 0, 0),
            datetime(2014, 6, 2, 0, 30),
            (0, 0),
            (1, 1)
        )

        self.assertListEqual(
            out_t,
            []
        )

        self.assertEqual(out_points.shape, (0, 2))

    def test_interpolate_slow(self):
        '''
        slower than expected
        '''
        vel = 1

        interpolator = SpeedAwareLinearInterpolator(
            vel, 
            15,
        )

        t0 = datetime(2014, 6, 2, 0, 0)
        t1 = datetime(2014, 6, 2, 1, 0)

        p0 = (0, 0)
        p1 = (1000, 1000)

        out_t, out_points = interpolator.interpolate(
            t0,
            t1,
            p0,
            p1
        )

        travel_time = timedelta(seconds=np.linalg.norm([1000, 1000], 2) / vel)
        begin_travel_time = t1 - travel_time

        self.assertListEqual(out_t, [
            begin_travel_time, begin_travel_time + timedelta(minutes=15)
        ])

        self.assertTrue(
            np.allclose(out_points, [
                [0, 0],
                [
                    np.interp(date_to_float(begin_travel_time + timedelta(minutes=15)), [date_to_float(begin_travel_time), date_to_float(t1)], [p0[0], p1[0]]),
                    np.interp(date_to_float(begin_travel_time + timedelta(minutes=15)), [date_to_float(begin_travel_time), date_to_float(t1)], [p0[1], p1[1]])
                ]
            ])
        )

    def test_interpolate_fast(self):
        '''
        faster than expected
        '''
        vel = 1e-10

        interpolator = SpeedAwareLinearInterpolator(
            vel, 
            15,
        )

        out_t, out_points = interpolator.interpolate(
            datetime(2014, 6, 2, 0, 0),
            datetime(2014, 6, 2, 0, 50),
            (0, 0),
            (1, 2)
        )

        self.assertListEqual(
            out_t,
            [datetime(2014, 6, 2, 0, 15), datetime(2014, 6, 2, 0, 30), datetime(2014, 6, 2, 0, 45)]
        )

        expected_out = np.stack([
            np.interp([15, 30, 45], [0, 50], [0, 1]),
            np.interp([15, 30, 45], [0, 50], [0, 2])
        ], axis=-1)

        self.assertTrue(np.allclose(
            out_points, expected_out
        ))
        



if __name__ == '__main__':
    unittest.main()