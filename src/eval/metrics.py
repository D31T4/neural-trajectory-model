'''
Implement evaluation metrics described in Trajectory Recovery from Ash
'''

import numpy as np
from src.data_preprocess.trajectory import Trajectory, to_numpy
from src.data_preprocess.point import Vec2
from src.utils import set_union, set_intersect, haversine

import heapq
from typing import Union, Literal

def top_k(trajectories: list[Trajectory], k: int) -> list[Vec2]:
    '''
    top k locations in a set of trajectories
    '''
    mset: dict[Vec2, int] = dict()

    for trajectory in trajectories:
        for point in trajectory:
            mset[point] = mset.get(point, 0) + 1

    return [tp[0] for tp in heapq.nlargest(k, mset.items(), key=lambda tp: tp[1])]

UniquenessMode = Union[Literal['all'], Literal['persistent']]

def uniqueness(trajectories: list[dict[str, Trajectory]], k: int, mode: UniquenessMode = 'all') -> float:
    '''
    compute uniqueness from Trajectory Reconstruction from Ash.

    This function is only used in exploratory analysis.

    Args:
    ---
    - trajectories
    - k: top-k most frequently visited locations to be considered
    - mode: type of users to be counted. `'all'` will count all users appeared. `'persistent'` will count users who are active in all time intervals.
    '''
    assert mode in ['all', 'persistent']

    users = None

    if mode == 'all':
        users = set_union(*[bucket.keys() for bucket in trajectories])
    else:
        users = set_intersect(*[bucket.keys() for bucket in trajectories])

    mset = dict()

    for uid in users:
        locs = top_k((t[uid] for t in trajectories if uid in t), k)
        locs = tuple(sorted(locs))

        mset[locs] = mset.get(locs, 0) + 1

    return sum(int(count == 1) for count in mset.values()) / len(users)

def uniqueness_list(trajectories: list[Trajectory], k: int) -> float:
    '''
    Compute uniqueness from a list of trajectories

    Args:
    ---
    - trajectories
    - k: top-k locations to be considered

    Returns:
    ---
    - % of unique trajectories
    '''
    mset = dict()

    for trajectory in trajectories:
        locs = [loc for loc, count in trajectory.k_most_frequently_visited(k)]
        locs = tuple(sorted(locs))

        mset[locs] = mset.get(locs, 0) + 1

    return sum(int(count == 1) for count in mset.values()) / len(trajectories)

def greedy_match(s0: list[Trajectory], s1: list[Trajectory]) -> list[int]:
    '''
    generate greedy matches for trajectories in s0 and s1

    Args:
    ---
    - s0: list of trajectories [n]
    - s1: list of trajectories [m]

    Returns:
    ---
    - indices of matched trajectories in s1 [n].
    '''
    a0 = to_numpy(s0)
    a1 = to_numpy(s1)

    s1_idx = [*range(len(s1))]
    s1_tail = len(s1)

    out = [-1] * len(s0)

    for i in range(len(s0)):
        if s1_tail == 0:
            break

        argmax = similarity(a0[i], a1[s1_idx[:s1_tail]]).argmax(axis=0)

        out[i] = s1_idx[argmax]

        # swap
        s1_idx[argmax] = s1_idx[s1_tail - 1]
        s1_tail -= 1

    return out

def similarity(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    '''
    compute accuracy metric from Trajectory Reconstruction from Ash

    Args:
    ---
    - s0: list of trajectories
    - s1: list of trajectories

    Returns:
    ---
    - average accuracy
    '''
    return (s0 == s1).prod(axis=-1).mean(axis=-1)

def mean_distance(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    '''
    pairwise mean distance between 2 batch of trajectories in lat-long

    Args:
    ---
    - s0: trajectory tensor [B, L, (lat, long)]
    - s1: trajectory tensor [B, L, (lat, long)]

    Returns:
    ---
    - distance in meter [B]
    '''
    x = haversine(s0, s1).mean(axis=-1)
    return x