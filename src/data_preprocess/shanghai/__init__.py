from datetime import timedelta
import pandas as pd
import tqdm
import math
import numpy as np

from src.data_preprocess.trajectory import Record, Trajectory, PreprocessConfig, aggregate_records, get_trajectories_one_day

def get_records(df: pd.DataFrame, config: PreprocessConfig) -> list[list[Record]]:
    '''
    get records from dataframe
    '''
    n_window = config.n_window()
    buckets: list[list[Record]] = [[] for _ in range(n_window)]

    time_delta = timedelta(minutes=config.delta_min)

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), disable=not config.verbose, desc='get_records'):
        # starting window
        j = math.floor((row['start time'] - config.start_date).total_seconds()) // (config.delta_min * 60)

        # window span
        span = math.floor((row['end time'] - (config.start_date + j * time_delta)).total_seconds()) // (config.delta_min * 60)
        
        start_time = max(row['start time'], config.start_date)

        j = max(0, j)
        l_bound = config.start_date + j * time_delta

        # fill spanning record
        for k in range(j, min(j + span + 1, n_window)):
            r_bound = l_bound + time_delta

            duration = (min(row['end time'], r_bound) - start_time).total_seconds() // 60
            duration = max(0, duration)

            buckets[k].append(Record(row['user id'], (row['latitude'], row['longitude']), duration))
            
            start_time = r_bound
            l_bound += time_delta

    return buckets


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df['start time'] = pd.to_datetime(df['start time'])
    df['end time'] = pd.to_datetime(df['end time'])

    return df

def preprocess(df: pd.DataFrame, config: PreprocessConfig) -> list[dict[str, Trajectory]]:
    # remove records with missing location
    df = df.dropna()

    # remove records not in Shanghai
    df = df[
        (120.9 <= df['longitude']) & (df['longitude'] <= 121.9) &
        (30.69 <= df['latitude']) & (df['latitude'] <= 31.51)
    ]

    # remove records not in time period
    df = df[
        (df['end time'] >= config.start_date) &
        (df['start time'] < config.start_date + timedelta(days=config.n_day))
    ]

    # discretize location
    df[['latitude', 'longitude']] = config.discretizer.discretize(df[['latitude', 'longitude']].to_numpy())

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
    from src.path import ROOT, default_shanghai_dataset_path
    from src.data_preprocess.point import NearestNeighborDiscretizer
    from src.data_preprocess.shanghai import PreprocessConfig, preprocess, read_csv

    from datetime import datetime

    discretizer = NearestNeighborDiscretizer(points=np.load(f'{ROOT}/exploratory_analysis/mog_50.npy'))

    config = PreprocessConfig(
        delta_min=30,
        start_date=datetime(2014, 6, 2),
        n_day=7,
        verbose=True
    )

    df = read_csv(f'{default_shanghai_dataset_path()}/data_6.1~6.15.csv')
    preprocess(df, config)

    print('dry run completed!')