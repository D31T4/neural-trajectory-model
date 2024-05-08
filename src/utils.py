from typing import TypeVar, Iterable

import numpy as np

T = TypeVar('T')

def set_union(*sets: Iterable[T]) -> set[T]:
    '''
    compute set union

    Args:
    ---
    - sets
    '''
    out = set()

    for s in sets:
        for el in s:
            out.add(el)

    return out

def set_intersect(*sets: Iterable[T]) -> set[T]:
    '''
    compute set intersect

    Args:
    ---
    - sets
    '''
    if len(sets) == 0:
        return set()

    out = set(sets[0])

    for i in range(1, len(sets)):
        out.intersection_update(sets[i])

    return out

def haversine(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    '''
    batch compute euclidean distance between 2 points (lat, long)

    Args:
    ---
    - p0: [b, (lat, long)]
    - p1: [b, (lat, long)]

    Returns:
    ---
    - euclidean distance (meter)
    '''
    R = 6371; # volumetric radius of earth (km)

    p0 = np.radians(p0)
    p1 = np.radians(p1)

    delta = p1 - p0

    # https://en.wikipedia.org/wiki/Haversine_formula
    a = np.sin(delta[..., 0] / 2) ** 2 + np.cos(p0[..., 0]) * np.cos(p1[..., 0]) * np.sin(delta[..., 1] / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    d = R * c
    return d * 1000