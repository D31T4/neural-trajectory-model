import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable

from src.ml.dataset import TrajectoryDataset
from src.ml.checkpoint import Checkpoint

class TrainConfig:
    '''
    train config
    '''
    
    def __init__(
        self,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        datasets: dict[str, TrajectoryDataset],
        n_epoch: int,
        all_candidates: torch.FloatTensor,
        verbose: bool = True,
        cuda: bool = False,
        checkpoint: Checkpoint = None,
        batch_size: int = 64,
        preprocess: Callable[[torch.FloatTensor], torch.FloatTensor] = None
    ):
        '''
        Args:
        ---
        - optimizer: optimizer
        - lr_scheduler: learning rate scheduler
        - datasets: dataset in dictionary with keys { 'train', 'valid', 'test' }
        - n_epoch: no. of epoch
        - all_candidates: candidate base stations
        - verbose: print output if true
        - cuda: train in cuda
        - checkpoint: checkpoint config
        - batch_size: batch size
        '''
        self.checkpoint = checkpoint or Checkpoint.none()

        self.verbose = verbose
        self.cuda = cuda
        self.n_epoch = n_epoch
        self.all_candidates = all_candidates
        self.batch_size = batch_size
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        #region dataset
        assert 'train' in datasets
        assert 'valid' in datasets
        self.datasets = {**datasets}

        if 'test' not in self.datasets:
            self.datasets['test'] = TrajectoryDataset.empty_dataset()
        #endregion

        if self.cuda:
            self.all_candidates = self.all_candidates.cuda()

        self.preprocess = preprocess

