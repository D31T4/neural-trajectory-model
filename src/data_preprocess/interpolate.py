import numpy as np
from datetime import datetime, timedelta
import time

from src.data_preprocess.point import Vec2

# temporal resolution epsilon
EPSILON = 0.1

class SpeedAwareLinearInterpolator:
    '''
    speed aware linear interpolation

    cases:
    ---
    - expected travel time > actual time delta: user is travelling for the entire period. `speed == travelled distance / actual time delta`
    - expected travel time < actual time delta: user is idle, then travel to end location with `speed == expected velocity`.
    '''

    def __init__(self, expected_speed: float, sample_interval: int):
        '''
        Args:
        ---
        - expected_speed: expected speed (meter per second)
        - sample_interval: sample interval (minute)
        '''
        self.expected_speed = expected_speed
        self.sample_interval = sample_interval

    def interpolate(self, t0: datetime, t1: datetime, p0: Vec2, p1: Vec2) -> tuple[list[datetime], np.ndarray]:
        '''
        interpolate points between (t0, p0) and (t1, p1)
        
        Args:
        ---
        - t0: start time
        - t1: end time
        - p0: start point
        - p1: end point

        Returns:
        ---
        - list of timestamps of added points
        - location of added points
        '''
        tspan_minute = (t1 - t0).total_seconds() / 60
        
        if self.expected_speed > 0:
            # expected travel time = distance / expected velocity
            t_travel_minute = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) / self.expected_speed / 60
        else:
            t_travel_minute = float('inf')

        all_points: list[np.ndarray] = [np.zeros((0, 2))] # placeholder
        all_t: list[datetime] = []

        # expected travel time longer than t1 - t0: t0 -> t1 == travel
        # add idle record
        if tspan_minute > t_travel_minute:
            all_t.append(t0 + timedelta(minutes=tspan_minute - t_travel_minute))

            static_points = np.array([[p0[0], p0[1]]])
            all_points.append(static_points)

        # add moving records
        start_t = t1 - timedelta(minutes=min(tspan_minute, t_travel_minute))

        x0 = time.mktime(start_t.timetuple())
        x1 = time.mktime(t1.timetuple())

        # subtract epsilon to prevent create t = t1
        dynamic_point_count = int(((t1 - start_t).total_seconds() - EPSILON) / (self.sample_interval * 60))
        
        if dynamic_point_count > 0:
            x = time.mktime(start_t.timetuple()) + (np.arange(dynamic_point_count) + 1) * self.sample_interval * 60

            all_t += [
                start_t + timedelta(minutes=self.sample_interval * (i + 1))
                for i in range(dynamic_point_count)
            ]

            dynamic_points = np.stack([
                np.interp(x, [x0, x1], [p0[0], p1[0]]),
                np.interp(x, [x0, x1], [p0[1], p1[1]])
            ], axis=-1)

            all_points.append(dynamic_points)

        all_points = np.concatenate(all_points, axis=0)
        return all_t, all_points

