import math
import numpy as np
import tqdm
import pandas as pd
from typing import Callable
from datetime import datetime
import itertools
import heapq

from src.data_preprocess.point import Vec2, Discretizer

class Record:
    def __init__(self, uid: str, loc: Vec2, votes: float):
        '''
        Args:
        ---
        - uid: user id
        - loc: base station
        - votes
        '''
        self.uid = uid
        self.loc = loc
        self.votes = votes

    def __eq__(self, r):
        if isinstance(r, Record):
            return self.uid == r.uid and self.loc == r.loc and self.votes == r.votes
        else:
            return False
        
    def __ne__(self, r):
        return not self.__eq__(r)
    
    def __repr__(self):
        return f'Record({self.uid}, {self.loc}, {self.votes})'
    
    def copy(self):
        return Record(self.uid, self.loc, self.votes)


class Trajectory:
    def __init__(self, points: list[Vec2]):
        '''
        Args:
        ---
        - points: points in trajectory
        '''
        assert len(points) > 0
        self.points = points

    def __len__(self):
        '''
        no. of points in trajectory
        '''
        return len(self.points)

    def __eq__(self, t) -> bool:
        '''
        compare trajectory
        
        Args:
        ---
        - t: trajectory
        '''
        if isinstance(t, Trajectory):
            return len(self) == len(t) and all(self.points[i] == t.points[i] for i in range(len(self)))
        else:
            return False
    
    def __ne__(self, t):
        '''
        compare trajectory

        Args:
        ---
        - t: trajectory
        '''
        return not self.__eq__(t)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, i: int) -> Vec2:
        return self.points[i]

    def copy(self):
        return Trajectory(self.points.copy())

    def completeness(self) -> float:
        return sum(int(p != None) for p in self.points) / len(self)
    
    def similarity(self, t: list[Vec2]) -> float:
        '''
        compute similarity between 2 trajectories.

        Args:
        ---
        - t: trajectory

        Returns:
        ---
        - similarity: `0 <= sim <= 1`
        '''
        assert len(self) == len(t)
        return sum(p0 == p1 for p0, p1 in zip(self, t)) / len(self)
    
    def visit_count(self):
        '''
        get visit count for each point in the trajectory

        Returns:
        ---
        - frequencies: point => visit count
        '''
        freq: dict[Vec2, int] = dict()

        for point in self:
            freq[point] = freq.get(point, 0) + 1

        return freq
    
    def k_most_frequently_visited(self, k: int):
        '''
        get k-most frequently visited locations

        Args:
        ---
        - k

        Reutrns:
        ---
        - list[(point, count)] with length at most k
        '''
        return [tp for tp in heapq.nlargest(k, self.visit_count().items(), key=lambda tp: tp[1])]

class TrajectoryBuilder(Trajectory):
    def __init__(self, points: list[Vec2] = None):
        '''
        Args:
        ---
        - points: initial points
        '''
        self.points = points or []

    def interpolate(self, method: Callable[[int, int, Vec2, Vec2], list[Vec2]]):
        '''
        interpolate and extrapolate missing records

        Args:
        ---
        - method: interpolation method

        Returns:
        ---
        - self
        '''
        assert len(self.points) > 0
        assert any(p != None for p in self.points)

        new_points = [p for p in self.points]

        prev_point_idx = -1

        for i in range(len(self.points)):
            if self.points[i] == None:
                continue

            if prev_point_idx == i - 1:
                # continuous
                prev_point_idx = i
                new_points[i] = self.points[i]

            else:
                if prev_point_idx == -1:
                    # extrapolate first segment
                    new_points[:(i + 1)] = [self.points[i]] * (i + 1)

                else:
                    new_points[(prev_point_idx + 1):i] = method(prev_point_idx, i, self.points[prev_point_idx], self.points[i])
                    new_points[i] = self.points[i]

                prev_point_idx = i

        if prev_point_idx != len(self.points) - 1:
            # extrapolate last
            new_points[(prev_point_idx + 1):] = [self.points[prev_point_idx]] * (len(self.points) - 1 - prev_point_idx)

        self.points = new_points
        return self
    
    @staticmethod
    def interp_nearest_fill(t0: int, t1: int, p0: Vec2, p1: Vec2) -> list[Vec2]:
        '''
        nearest fill
        '''
        mid = math.floor((t0 + t1) / 2)
        return [p0] * max(mid - t0, 0) + [p1] * max(t1 - mid - 1, 0)

    @staticmethod
    def interp_ffill(t0: int, t1: int, p0: Vec2, p1: Vec2) -> list[Vec2]:
        '''
        forward fill
        '''
        return [p0] * max(t1 - t0 - 1, 0)
    
    @staticmethod
    def interp_bfill(t0: int, t1: int, p0: Vec2, p1: Vec2) -> list[Vec2]:
        '''
        backward fill
        '''
        return [p1] * max(t1 - t0 - 1, 0)
    
    def build(self) -> Trajectory:
        '''
        build trajectory

        Returns:
        ---
        - trajectory
        '''
        return Trajectory(tuple(self.points))

class PreprocessConfig:
    '''
    pre-process config
    '''
    
    def __init__(
        self, 
        delta_min: int, 
        start_date: datetime, 
        n_day: int, 
        verbose: bool = False,
        interp_trajectory: bool = True,
        keep_trajectory: Callable[[Trajectory], bool] = lambda _: True,
        discretizer: Discretizer = None
    ):
        '''
        Args
        ---
        - delta_min: temporal discretization window width in minute
        - start_date: start date of dataframe
        - n_day: no. of days in dataframe
        - verbose: print message if set to `True`
        - interp_trajectory
        - keep_trajectory: keep trajectory in `get_trajectory` if set to `True`, runs before interpolation
        - discretizer: discretization scheme. No discretization if `None`
        '''
        
        # divisibility check
        assert (24 * 60) % delta_min == 0

        # window width in minute
        self.delta_min = delta_min
        
        # no. of window per day
        self.n_window_per_day = (24 * 60) // self.delta_min

        # no. of day
        self.n_day = n_day

        self.start_date = start_date

        self.verbose = verbose

        self.interp_trajectory = interp_trajectory
        self.keep_trajectory = keep_trajectory

        self.discretizer = discretizer or Discretizer()

    def n_window(self) -> int:
        '''
        no. of windows
        '''
        return self.n_day * self.n_window_per_day


def aggregate_records(buckets: list[list[Record]], config: PreprocessConfig) -> list[list[Record]]:
    '''
    select most frequent location for each user in each bucket

    Args:
    ---
    - buckets: buckets of records
    - config

    Returns:
    ---
    - bucket of points: In each bucket, users are now distinct. And records only contains most frequently visited location.
    '''
    new_buckets: list[list[Record]] = [[] for _ in buckets]
    
    for i in tqdm.trange(len(buckets), disable=not config.verbose, desc='aggregate_records'):
        bucket = sorted(buckets[i], key=lambda el: (el.uid, el.loc))
        j = 0

        # repeat for each user
        while j < len(bucket):
            # find location with longest total duration of user
            curr_uid = bucket[j].uid
            curr_loc = bucket[j].loc
            votes_curr_loc = bucket[j].votes

            max_loc = curr_loc
            votes_max_loc = votes_curr_loc

            j += 1
            while j < len(bucket) and bucket[j].uid == curr_uid:
                if bucket[j].loc != curr_loc:
                    if votes_curr_loc > votes_max_loc:
                        votes_max_loc = votes_curr_loc
                        max_loc = curr_loc

                    curr_loc = bucket[j].loc
                    votes_curr_loc = bucket[j].votes
                else:
                    votes_curr_loc += bucket[j].votes

                j += 1

            # update last cluster
            if votes_curr_loc > votes_max_loc:
                votes_max_loc = votes_curr_loc
                max_loc = curr_loc

            new_buckets[i].append(Record(curr_uid, max_loc, votes_max_loc))

    return new_buckets

def get_trajectories_one_day(buckets: list[list[Record]], config: PreprocessConfig) -> dict[str, Trajectory]:
    '''
    get trajectories

    Args:
    ---
    - buckets: buckets of points. expected length == no. of window per day
    - config

    Returns:
    ---
    - map: user id => trajectory
    '''
    assert len(buckets) == config.n_window_per_day

    trajectories = { r.uid: TrajectoryBuilder([None] * config.n_window_per_day) for r in itertools.chain(*buckets) }

    # build trajectory
    for i in range(config.n_window_per_day):
        for record in buckets[i]:
            trajectories[record.uid].points[i] = record.loc

    # keys to be removed
    to_remove: list[str] = []

    for key, trajectory in trajectories.items():
        if not config.keep_trajectory(trajectory):
            to_remove.append(key)
            continue

        if config.interp_trajectory:
            trajectory.interpolate(TrajectoryBuilder.interp_ffill)

    # delete items
    for key in to_remove:
        del trajectories[key]
    
    return trajectories


def to_dataframe(trajectories: dict[str, Trajectory], trajectory_len: int) -> pd.DataFrame:
    '''
    convert a list of trajectories to a pandas dataframe
    '''
    assert all(len(t) == trajectory_len for t in trajectories.values())

    records = []

    for uid, trajectory in trajectories.items():
        records += [
            (uid, j, trajectory[j][0], trajectory[j][1])
            for j in range(trajectory_len)
            if trajectory[j] != None
        ]

    return pd.DataFrame.from_records(
        records,
        columns=['uid', 't', 'lat', 'long']
    )

def from_dataframe(df: pd.DataFrame, trajectory_len: int):
    '''
    create list of trajectories from a pandas dataframe
    '''
    trajectories: dict[str, Trajectory] = dict()

    for _, row in df.iterrows():
        uid = row['uid']

        t = trajectories.get(uid, None)

        if t == None:
            t = Trajectory([None] * trajectory_len)
            trajectories[uid] = t

        t.points[row['t']] = (row['lat'], row['long'])

    assert all(
        sum(int(p != None) for p in t) == trajectory_len 
        for t in trajectories.values()
    )

    return trajectories

def from_numpy(array: np.ndarray) -> list[Trajectory]:
    '''
    convert a numpy array to a list of trajectories

    Args:
    ---
    - array: [B, L, (lat, long)]

    Returns:
    ---
    - a list of trajectories
    '''
    trajectories = [Trajectory([None] * array.shape[1]) for _ in range(array.shape[0])]

    for i in range(array.shape[0]):
        for t in range(array.shape[1]):
            trajectories[i].points[t] = tuple(array[i, t].tolist())

    return trajectories

def to_numpy(trajectories: list[Trajectory]) -> np.ndarray:
    '''
    convert a list of trajectories into a numpy array

    Returns:
    ---
    - trajectory tensor
    '''
    return np.array(trajectories)