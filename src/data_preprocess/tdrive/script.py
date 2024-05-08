import os
import tqdm
from datetime import datetime
from typing import Callable

from src.data_preprocess.tdrive import preprocess, read_txt, TDrivePreprocessConfig
from src.data_preprocess.trajectory import Trajectory
from src.data_preprocess.point import Discretizer

from src.path import default_tdrive_dataset_path

class ScriptConfig:
    def __init__(
        self,
        delta_min: int,
        out_dir: str,
        discretizer: Discretizer,
        verbose: bool = True,
        interp_trajectory: bool = True,
        keep_trajectory: Callable[[Trajectory], bool] = lambda _: True
    ):
        self.discretizer = discretizer

        self.delta_min = delta_min
        self.n_day = 7
        self.verbose = verbose
        self.interp_trajectory = interp_trajectory
        self.keep_trajectory = keep_trajectory

        self.out_dir = out_dir

    def build(self) -> TDrivePreprocessConfig:
        return TDrivePreprocessConfig(
            delta_min=self.delta_min,
            start_date=datetime(2008, 2, 4),
            n_day=self.n_day,
            verbose=False,
            interp_trajectory=self.interp_trajectory,
            keep_trajectory=self.keep_trajectory
        )

def run(config: ScriptConfig):
    dataset_path = default_tdrive_dataset_path()

    # create csv
    for i in range(config.n_day):
        pass

    for fname in tqdm.tqdm(os.listdir(dataset_path)[:10]):
        df = read_txt(f'{dataset_path}/{fname}')

        trajectories = preprocess(df, config.build())

        for i in range(config.n_day):
            pass

            # write csv
            for point in trajectories[i]:
                pass

if __name__ == '__main__':
    raise NotImplementedError()
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--delta', type=int, default=30)
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('-s', '--silent', action='store_true')

    args = parser.parse_args()

    config = ScriptConfig(
        delta_min=args.delta,
        out_dir=args.dir,
        verbose=not args.silent,
        discretizer=Discretizer()
    )

    if os.path.exists(config.out_dir):
        raise FileExistsError('directory already exists!')
        
    os.makedirs(config.out_dir)

    run(config)