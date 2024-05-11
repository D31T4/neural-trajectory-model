import numpy as np
import pandas as pd
import tqdm
import math
from datetime import datetime, timedelta
from typing import Callable

from src.data_preprocess.trajectory import PreprocessConfig, aggregate_records, get_trajectories_one_day
from src.data_preprocess.trajectory import Record, Trajectory
from src.data_preprocess.interpolate import SpeedAwareLinearInterpolator
from src.data_preprocess.point import LatLong, CoordinateMap, Discretizer
from src.utils import haversine

class TDrivePreprocessConfig(PreprocessConfig):
    def __init__(
        self,
        delta_min: int, 
        start_date: datetime, 
        n_day: int, 
        verbose: bool = False,
        interp_trajectory: bool = True,
        keep_trajectory: Callable[[Trajectory], bool] = lambda _: True,
        discretizer: Discretizer = None,
        expected_speed: float = 40e3 / 3600
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
        - expected_speed: expected speed of the user. compute average from dataframe if set to `None`
        '''
        assert expected_speed == None or expected_speed >= 0

        super().__init__(
            delta_min, 
            start_date, 
            n_day, 
            verbose,
            interp_trajectory,
            keep_trajectory,
            discretizer
        )

        self.expected_speed = expected_speed
        self.uid: int = None

def read_txt(fname: str) -> pd.DataFrame:
    '''
    read t-drive dataset txt files

    Args:
    ---
    - fname: file name

    Returns:
    ---
    - dataframe
    '''
    df = pd.read_csv(fname, sep=',', header=None, names=['uid', 't', 'long', 'lat'])
    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')

    return df

def get_records(df: pd.DataFrame, config: TDrivePreprocessConfig) -> list[list[Record]]:
    '''
    get records from dataframe

    Args:
    ---
    - df: dataframe
    '''
    assert len(df) > 0

    n_window = config.n_window()
    buckets: list[list[Record]] = [[] for _ in range(n_window)]

    time_delta = timedelta(minutes=config.delta_min)

    prev_t: datetime = None
    prev_loc: LatLong = None

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), disable=not config.verbose, desc='get_records'):
        if prev_t == None:
            prev_t = config.start_date
            prev_loc = (row['lat'], row['long'])

        # starting window
        j: int = math.floor((prev_t - config.start_date).total_seconds()) // (config.delta_min * 60)

        # window span
        span = math.floor((row['t'] - (config.start_date + j * time_delta)).total_seconds()) // (config.delta_min * 60)

        start_time = max(prev_t, config.start_date)

        j = max(0, j)
        l_bound = config.start_date + j * time_delta

        # fill spanning record
        for k in range(j, min(j + span + 1, n_window)):
            r_bound = l_bound + time_delta

            duration = (min(row['t'], r_bound) - start_time).total_seconds() // 60
            duration = max(0, duration)

            buckets[k].append(Record(row['uid'], prev_loc, duration))
            
            start_time = r_bound
            l_bound += time_delta

        prev_t = row['t']
        prev_loc = (row['lat'], row['long'])
            
    return buckets

def average_speed(df: pd.DataFrame):
    '''
    compute average speed.

    Args:
    ---
    - dataframe

    Returns:
    ---
    - expected speed
    '''
    loc = df[['lat', 'long']].to_numpy()
    time = pd.to_datetime(df['t']).astype(np.int64).to_numpy() / 1e9

    dx = haversine(loc[1:], loc[:-1])
    dt = time[1:] - time[:-1]

    # remove dt = 0 due to dupliacted time stamp
    dx = dx[dt > 0]
    dt = dt[dt > 0]

    # remove stationary periods
    dt = dt[dx > 100]
    dx = dx[dx > 100]

    return np.mean(dx / dt)

def interpolate(df: pd.DataFrame, config: TDrivePreprocessConfig) -> pd.DataFrame:
    '''
    interpolate and extrapolate missing points

    cases:
    ---
    - interpolation: `SpeedAwareInterpolator`
    - extrapolation: use last known position

    Args:
    ---
    - uid
    - df: dataframe
    - config: configuration

    Returns:
    ---
    - sorted dataframe with interpolated points
    '''
    df = df.sort_values(by=['t'])

    coord_map = CoordinateMap(ref=(39.915, 116.395))

    # calculate expected speed. not accurate :(
    expected_vel = config.expected_speed if config.expected_speed != None else average_speed(df)

    # average sampling interval is around 177 seconds
    sample_interval = 3

    interpolator = SpeedAwareLinearInterpolator(
        expected_speed=expected_vel,
        sample_interval=sample_interval
    )

    # interpolate if needed
    prev_t: datetime = config.start_date
    prev_loc: LatLong = None

    new_points = []

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), disable=not config.verbose, desc='continuous interpolation'):
        curr_loc = (row['lat'], row['long'])

        if prev_loc == None:
            prev_loc = curr_loc

        delta_t: timedelta = row['t'] - prev_t

        # same location, assume stationary
        if prev_loc == curr_loc:
            prev_t = row['t']
            continue


        if delta_t.total_seconds() <= sample_interval * 60:
            continue
        
        p0, p1 = coord_map.to_cartesian(np.array([
            [prev_loc[0], prev_loc[1]],
            [curr_loc[0], curr_loc[1]]
        ]))

        out_t, out_points = interpolator.interpolate(
            prev_t, 
            row['t'].to_pydatetime(),
            (p0[0], p0[1]),
            (p1[0], p1[1])
        )

        out_points = coord_map.to_latlong(out_points)

        new_points += [
            (config.uid, out_t[j], out_points[j, 1], out_points[j, 0])
            for j in range(len(out_t))
        ]
        
        prev_t = row['t']
        prev_loc = curr_loc

    if prev_loc != None:
        # add final point
        new_points.append(
            (config.uid, config.start_date + timedelta(days=config.n_day), prev_loc[1], prev_loc[0])
        )
        
    new_points_df = pd.DataFrame.from_records(new_points, columns=['uid', 't', 'long', 'lat'])

    df = pd.concat([df, new_points_df])
    df = df.sort_values(by=['t'])
    
    return df

def preprocess(df: pd.DataFrame, config: TDrivePreprocessConfig) -> list[Trajectory]:
    # filter df
    df = df[
        (39.75 <= df['lat']) & (df['lat'] <= 40.026) &
        (116.2 <= df['long']) & (df['long'] <= 116.55)
    ]

    if len(df) == 0:
        return []

    # interpolate
    df = interpolate(df, config)

    # discretize location
    df[['lat', 'long']] = config.discretizer.discretize(df[['lat', 'long']].to_numpy())

    buckets = get_records(df, config)
    buckets = aggregate_records(buckets, config)

    out = []

    # get trajectory
    for head in range(0, len(buckets), config.n_window_per_day):
        trajectories = get_trajectories_one_day(buckets[head:(head + config.n_window_per_day)], config)
        out.append(trajectories)

    return out

if __name__ == '__main__':
    '''
    dry run
    '''
    from src.data_preprocess.point import NearestNeighborDiscretizer
    from src.path import default_tdrive_dataset_path, ROOT

    discretizer = NearestNeighborDiscretizer(np.load(f'{ROOT}/exploratory_analysis/tdrive_mog100.npy'))

    config = TDrivePreprocessConfig(
        delta_min=30,
        start_date=datetime(2008, 2, 4),
        n_day=7,
        verbose=True,
        interp_trajectory=True,
        discretizer=discretizer
    )

    config.uid = 1

    preprocess(
        read_txt(f'{default_tdrive_dataset_path()}/1.txt'), 
        config
    )

    print('dry run completed!')