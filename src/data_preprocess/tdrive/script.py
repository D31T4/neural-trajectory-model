import os
import shutil
import tqdm
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocess.tdrive import preprocess, TDrivePreprocessConfig
from src.data_preprocess.trajectory import Trajectory, to_dataframe
from src.data_preprocess.point import Discretizer, NearestNeighborDiscretizer

from src.path import default_tdrive_dataset_path

class ScriptConfig:
    def __init__(
        self,
        delta_min: int,
        out_dir: str,
        discretizer: Discretizer,
        verbose: bool = True,
        interp_trajectory: bool = True,
        keep_trajectory: Callable[[Trajectory], bool] = lambda _: True,
        user_list: list[str] = None
    ):
        self.discretizer = discretizer

        self.delta_min = delta_min
        self.start_date = datetime(2008, 2, 2)
        self.n_day = 7
        self.verbose = verbose
        self.interp_trajectory = interp_trajectory
        self.keep_trajectory = keep_trajectory

        self.out_dir = out_dir
        
        self.user_list = user_list

    def build(self, uid: str) -> TDrivePreprocessConfig:
        return TDrivePreprocessConfig(
            uid=uid,
            delta_min=self.delta_min,
            start_date=self.start_date,
            n_day=self.n_day,
            verbose=False,
            interp_trajectory=self.interp_trajectory,
            keep_trajectory=self.keep_trajectory
        )
    
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
    df = pd.read_csv(
        fname, 
        sep=',', 
        header=None, 
        names=['uid', 't', 'long', 'lat'],
        dtype={ 'uid': str, 't': str, 'lat': np.float64, 'long': np.float64 }
    )

    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')

    return df

def run(config: ScriptConfig):
    dataset_path = default_tdrive_dataset_path()
    file_list = config.user_list or [fname.split('.')[0] for fname in os.listdir(dataset_path)][:10]

    for uid in tqdm.tqdm(file_list, desc='pre-process', disable=not config.verbose):
        df = read_txt(f'{dataset_path}/{uid}.txt')

        proc_config = config.build(uid)
        trajectories = preprocess(df, proc_config)

        if len(trajectories) == 0:
            continue

        for i in range(config.n_day):
            current_date = config.start_date + timedelta(days=i)
            fname = f'{config.out_dir}/{current_date.year}-{current_date.month}-{current_date.day}.csv'

            # write csv
            assert uid in trajectories[i]
            df = to_dataframe({ uid: trajectories[i][uid] }, trajectory_len=proc_config.n_window_per_day)
            df.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--delta', type=int, default=30, help='discretization granularity')
    parser.add_argument('--dir', type=str, required=True, help='output directory')
    parser.add_argument('--cluster_path', type=str, required=True, help='path to pre-computed clusters; expected to be a saved numpy array.')
    parser.add_argument('-s', '--silent', action='store_true', help='verbosity')
    parser.add_argument('--partition', type=str, choices=['train', 'test'], help='data partition')

    args = parser.parse_args()

    config = ScriptConfig(
        delta_min=args.delta,
        out_dir=args.dir,
        verbose=not args.silent,
        discretizer=NearestNeighborDiscretizer(np.load(args.cluster_path))
    )

    if os.path.exists(config.out_dir):
        raise FileExistsError('directory already exists!')

    os.makedirs(config.out_dir)

    # train test split
    user_list = [fname.split('.')[0] for fname in os.listdir(default_tdrive_dataset_path())]
    train_list, test_list = train_test_split(user_list, test_size=0.2, random_state=123)
    config.user_list = train_list if args.partition == 'train' else test_list

    run(config)