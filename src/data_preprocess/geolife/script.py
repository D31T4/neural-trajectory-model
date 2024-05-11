import os
import tqdm
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime, timedelta

from src.data_preprocess.point import Discretizer, NearestNeighborDiscretizer
from src.data_preprocess.geolife import preprocess, GeoLifePreprocessConfig

from src.data_preprocess.trajectory import to_dataframe
from src.path import default_geolife_dataset_path

class ScriptConfig:
    def __init__(
        self,
        delta_min: int,
        out_dir: str,
        discretizer: Discretizer,
        verbose: bool = False
    ):
        self.delta_min = delta_min
        self.out_dir = out_dir
        self.discretizer = discretizer
        self.verbose = verbose
    

def read_one_file(fname: str) -> pd.DataFrame:
    '''
    read one data file

    Args:
    ---
    - fname: filename

    Returns:
    ---
    - dataframe
    '''
    df = pd.read_csv(
        fname, 
        skiprows=6, 
        names=['lat', 'long', '0', 'alt', 'day', 'date', 'time'], 
        dtype={ 'lat': np.float64, 'long': np.float64, '0': np.int32, 'alt': np.float64, 'day': np.float64, 'date': str, 'time': str }
    )[['date', 'time', 'lat', 'long']]

    df['date'] += ' ' + df['time']
    del df['time']

    df = df.dropna()
    
    return df


def read_data(folder: str) -> pd.DataFrame:
    '''
    read one user data
    '''
    with tempfile.TemporaryFile() as tf:
        for fname in os.listdir(folder):
            # write to temp file
            df = read_one_file(f'{folder}/{fname}')
            df.to_csv(tf, mode='a', header=False, index=False)

        # read and return temp file
        tf.seek(0) # reset cursor

        return pd.read_csv(
            tf,
            names=['t', 'lat', 'long'],
            dtype={ 't': str, 'lat': np.float64, 'long': np.float64 }
        )


def run_preprocess(config: ScriptConfig):
    in_dir = default_geolife_dataset_path()
    folders = os.listdir(in_dir)

    for uid in tqdm.tqdm(folders, desc='pre-process', disable=not config.verbose):
        df = read_data(f'{in_dir}/{uid}/Trajectory')
        df['uid'] = uid
        df['t'] = pd.to_datetime(df['t'])

        start_date: datetime = df['t'].min().to_pydatetime()
        n_day: int = (df['t'].max().to_pydatetime() - start_date).days + 1

        proc_config = GeoLifePreprocessConfig(
            uid=uid,
            delta_min=config.delta_min,
            start_date=start_date,
            n_day=n_day,
            discretizer=config.discretizer,
            expected_speed=None,
            verbose=False
        )

        trajectories = preprocess(df, proc_config)

        # save to csv
        current_date = start_date

        for trajectory in trajectories:
            if uid in trajectory:
                fname = f'{config.out_dir}/{current_date.year}-{current_date.month}-{current_date.day}.csv'

                df = to_dataframe({ uid: trajectory[uid] }, proc_config.n_window_per_day)
                df.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)

            current_date += timedelta(days=1)
            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='run pre-process script of GeoLife dataset'
    )

    parser.add_argument('--delta', type=int, default=30, help='discretization granularity')
    parser.add_argument('--dir', type=str, required=True, help='output directory')
    parser.add_argument('--cluster_path', type=str, required=True, help='path to pre-computed clusters; expected to be a saved numpy array.')
    parser.add_argument('-s', '--silent', action='store_true', help='verbosity')

    args = parser.parse_args()

    config = ScriptConfig(
        delta_min=args.delta,
        out_dir=args.dir,
        discretizer=NearestNeighborDiscretizer(np.load(args.cluster_path)),
        verbose=not args.silent
    )

    if os.path.exists(config.out_dir):
        raise FileExistsError('directory already exists!')
    
    os.makedirs(config.out_dir)

    run_preprocess(config)

