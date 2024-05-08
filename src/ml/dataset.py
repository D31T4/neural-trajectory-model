'''
dataset for torch
'''
import tqdm
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from datetime import date

from typing import Callable

from src.data_preprocess.point import Vec2
from src.data_preprocess.trajectory import Trajectory

CACHE_PATH = str(Path(__file__).parent.absolute().joinpath('cache'))

if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

def hour_sequence(trajectory_len: int) -> list[int]:
    '''
    generate list of hour for each timestamp in a trajectory

    Args:
    ---
    - trajectory

    Returns:
    ---
    - sequence of hour
    '''
    step = (24 * 60) // trajectory_len
    assert step * trajectory_len == 24 * 60

    return [
        (i * step) // 60
        for i in range(trajectory_len)
    ]

def get_shanghai_date(fname: str) -> date:
    '''
    get date from shanghai dataset filename

    Args:
    ---
    - fname: file name

    Returns:
    ---
    - date
    '''
    fname = fname[:fname.index('.')]
    month, day = fname.split('-')
    return date(2014, int(month), int(day))

def trajectory_to_tensor(trajectory: Trajectory) -> torch.FloatTensor:
    '''
    convert trajectory to tensor

    Args:
    ---
    - trajectory

    Returns:
    ---
    - tensor[L, 2]
    '''
    out = torch.zeros((len(trajectory), 2), dtype=torch.float32)

    for i in range(len(trajectory)):
        out[i, 0] = trajectory[i][0]
        out[i, 1] = trajectory[i][1]

    return out

def create_point_to_class_map(candidates: torch.FloatTensor) -> dict[Vec2, int]:
    '''
    create mapping from point to index

    Args:
    ---
    - points

    Returns:
    ---
    - dict
    '''
    map: dict[Vec2, int] = dict()

    for i in range(candidates.shape[0]):
        point = tuple(candidates[i].tolist())
        map[point] = i

    return map

DatasetEntry = tuple[torch.FloatTensor, torch.IntTensor]

class TrajectoryDataset(Dataset):
    '''
    trajectory dataset
    '''
    
    def __init__(
        self, 
        sequence_length: int, 
        point_to_class_map: dict[Vec2, int]
    ):
        '''
        Args:
        ---
        - sequence_length: trajectory length
        - point_to_class_map
        '''
        self.sequence_length = sequence_length
        self.trajectories: list[tuple[torch.FloatTensor, int]] = []
        self.file_idx = [0]
        self.point_to_class_map = point_to_class_map

    @staticmethod
    def empty_dataset():
        '''
        create empty dataset instance
        '''
        return TrajectoryDataset(0, dict())

    def read_files(
        self, 
        files: list[str],
        read_file: Callable[[str], tuple[date, list[Trajectory]]],
        verbose: bool = True
    ):
        '''
        load data files in memory

        Args:
        ---
        - files: list of name of files to be consumed
        - read_file: consume file, returns date and list of trajectories
        - verbose: show loading progress
        '''
        for fname in tqdm.tqdm(files, desc='loading dataset', disable=not verbose):
            day, trajectories = read_file(fname)
            day = day.weekday()

            out: list[tuple[torch.FloatTensor, int]] = [None] * len(trajectories)

            for i, trajectory in enumerate(trajectories):
                trajectory = trajectory_to_tensor(trajectory)
                out[i] = (trajectory, day)

            self.trajectories += out
            self.file_idx.append(len(self.trajectories))

    def load(self, path: str):
        self.trajectories, self.file_idx = torch.load(path)

    def save(self, path):
        torch.save([
            self.trajectories,
            self.file_idx
        ], path)

    def __len__(self) -> int:
        '''
        no. of trajectories
        '''
        return len(self.trajectories)
    
    def get_trajectory_class(self, trajectory: torch.FloatTensor):
        '''
        get trajectory class tensor

        Args:
        ---
        - trajectory

        Returns:
        ---
        - class tensor [L]
        '''
        classes = torch.zeros(self.sequence_length, dtype=int)

        for l in range(self.sequence_length):
            # convert tensor to tuple
            loc = tuple(trajectory[l].tolist())
            classes[l] = self.point_to_class_map[loc]

        return classes

    def __getitem__(self, idx: int):
        '''
        get item
        '''
        trajectory, weekday = self.trajectories[idx]

        # prepare context vector
        hour = F.one_hot(
            torch.tensor(hour_sequence(self.sequence_length), dtype=int), 
            24
        )

        weekday = F.one_hot(
            torch.tensor(weekday, dtype=int),
            7
        ).repeat((self.sequence_length, 1))

        # context tensor
        context = torch.cat((hour, weekday), dim=-1).to(torch.float32)

        targets = self.get_trajectory_class(trajectory)

        return trajectory, context, targets
