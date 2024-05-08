import os
import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable

from src.data_preprocess.shanghai import read_csv, preprocess
from src.data_preprocess.trajectory import to_dataframe, Trajectory, PreprocessConfig
from src.data_preprocess.point import Discretizer, NearestNeighborDiscretizer
from src.path import default_shanghai_dataset_path, ROOT

import argparse

def extract_date_range(fname: str):
    '''
    extract date range from file name

    Args:
    ---
    - fname: file name


    Returns:
    ---
    - min date
    - max date
    '''
    fname = fname[:fname.rindex('.')] # remove extension
    fname = fname[5:] # remove prefix
    
    start_date, end_date = fname.split('~')

    return (
        datetime(2014, int(start_date.split('.')[0]), int(start_date.split('.')[1])), 
        datetime(2014, int(end_date.split('.')[0]), int(end_date.split('.')[1]))
    )

class ScriptConfig:
    def __init__(
        self,
        delta_min: int,
        out_dir: str,
        discretizer: Discretizer,
        verbose: bool = False,
        interp_trajectory: bool = True,
        keep_trajectory: Callable[[Trajectory], bool] = lambda _: True
    ):
        '''
        script config

        Args:
        ---
        - delta_min: temporal discretization window width
        - out_dir: output directory
        - discretizer
        - verbose: show progress if set to `True`
        - interp_trajectory: interpolate missing points in trajectory
        - keep_trajectory: remove trajectory if return `False`
        '''
        self.delta_min = delta_min
        self.out_dir = out_dir
        self.discretizer = discretizer
        self.verbose = verbose
        self.interp_trajectory = interp_trajectory
        self.keep_trajectory = keep_trajectory

    def build(
        self,
        start_date: datetime,
        n_day: int
    ) -> PreprocessConfig:
        '''
        factory method for pre-process config
        '''
        return PreprocessConfig(
            self.delta_min,
            start_date,
            n_day,
            self.verbose,
            self.interp_trajectory,
            self.keep_trajectory,
            discretizer=self.discretizer
        )

def run_preprocess(config: ScriptConfig):
    '''
    run pre-process script

    Args:
    ---
    - config
    '''
    dataset_path = default_shanghai_dataset_path()


    # sort by file start date
    fnames = sorted(os.listdir(dataset_path), key=lambda fname: float(fname.split('~')[0][5:]))

    for i in range(len(fnames)):
        df = read_csv(f'{dataset_path}/{fnames[i]}')

        # append records with end date past midnight of last day
        min_date, max_date = extract_date_range(fnames[i])
        n_day = max_date.day - min_date.day + 1

        if i + 1 < len(fnames):
            df2 = read_csv(f'{dataset_path}/{fnames[i + 1]}')
            df2 = df2[df2['start time'] < min_date + timedelta(days=n_day)]
            df = pd.concat([df, df2])

        trajectories = preprocess(
            df, 
            config.build(min_date, n_day)
        )

        #region save daily trajectory as csv
        curr_date = min_date

        for bucket in trajectories:
            out = to_dataframe(bucket, 24 * 60 // config.delta_min)

            fout = f'{curr_date.month}-{curr_date.day}.csv'
            out.to_csv(f'{config.out_dir}/{fout}', index=False)

            curr_date += timedelta(days=1)
        #endregion

        if config.verbose:
            print(f'{fnames[i]} done!')

        
def run_csv(dir: str):
    '''
    convert xlsx data files to csv for faster read
    
    Args:
    ---
    - dir: input dir
    '''
    out_dir = default_shanghai_dataset_path()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fname in tqdm.tqdm(os.listdir(dir)):
        df = pd.read_excel(f'{dir}/{fname}')
        df = df.dropna()

        fname = fname[:fname.rindex('.')]

        df = df.sort_values(by=['month', 'date', 'start time'])

        df.to_csv(f'{out_dir}/{fname}.csv', index=False)

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='run pre-process script of Shanghai Telecom dataset'
    )

    parser.add_argument('method', type=str, choices=['csv', 'proc'], help='process mode. proc: pre-process; csv: convert xlsx to csv')
    parser.add_argument('--delta', type=int, default=30, help='discretization granularity')
    parser.add_argument('--dir', type=str, required=True, help='output directory if method = proc; input directory if method = csv')
    parser.add_argument('--cluster_path', type=str, required=False, help='path to pre-computed clusters; expected to be a saved numpy array. put `false` if you do not want spatial aggregation.')
    parser.add_argument('-s', '--silent', action='store_true', help='verbosity')

    args = parser.parse_args()

    if args.method == 'proc':
        if not args.cluster_path:
            cluster_path = f'{ROOT}/exploratory_analysis/mog_100.npy'
        else:
            cluster_path: str = args.cluster_path

        if cluster_path == 'false':
            discretizer = Discretizer()
        else:
            discretizer = NearestNeighborDiscretizer(points=np.load(cluster_path))

        config = ScriptConfig(
            delta_min=args.delta,
            out_dir=args.dir,
            discretizer=discretizer,
            verbose=not args.silent
        )

        if os.path.exists(config.out_dir):
            raise FileExistsError('directory already exists!')
        
        os.makedirs(config.out_dir)

        run_preprocess(config)
    else:
        run_csv(args.dir)
