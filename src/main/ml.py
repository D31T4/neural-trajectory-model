import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.dataset import hour_sequence
from src.ml.model import TrajectoryModel
from src.main.baseline import DailyTrajectoryRecovery, GetCostMatrix

from typing import Callable

def create_trajectory_model_cost_matrix_fn(
    model: TrajectoryModel, 
    preprocess: Callable[[torch.FloatTensor], torch.FloatTensor], 
    weekday: int, 
    trajectory_len: int,
    device: torch.device
) -> GetCostMatrix:
    '''
    factory method for creating `cost_matrix` argument used in `DailyTrajectoryRecovery`

    Args:
    ---
    - model: trained trajectory model
    - preprocess: preprocess pipeline for basestation
    - weekday: an integer in [0-6] denoting the day of week
    - trajectory_len: trajectory length
    - device: device of model

    Returns:
    ---
    - cost matrix
    '''
    week_context = F.one_hot(torch.tensor(weekday, dtype=int), 7)
    hour_context = F.one_hot(
        torch.tensor(hour_sequence(trajectory_len), dtype=int),
        24
    )

    context = torch.concat((
        hour_context,
        week_context.unsqueeze(0).repeat(trajectory_len, 1)
    ), dim=-1).to(dtype=torch.float32, device=device)

    def cost_matrix_fn(trajectories: np.ndarray, candidates: np.ndarray, tstamp: int):
        trajectories = preprocess(torch.tensor(trajectories, dtype=torch.float32, device=device))
        candidates = preprocess(torch.tensor(candidates, dtype=torch.float32, device=device))

        likelihood = model.cost_matrix(
            context[:tstamp],
            trajectories,
            candidates,
            context[tstamp]
        )

        return likelihood.cpu().numpy()

    return cost_matrix_fn