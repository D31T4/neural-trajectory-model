import torch
from torch.utils.data.dataloader import DataLoader

from src.ml.model import TrajectoryModel
from src.ml.config import TrainConfig
from src.ml.stats import TrainStatistics, ValidationStatistics, TestStatistics

class TrainState:
    '''
    train state
    '''
    
    def __init__(
        self,
        model: TrajectoryModel,
        config: TrainConfig
    ):
        '''
        Args:
        ---
        - model
        - config: train config
        '''
        self.current_epoch: int = 0

        self.model = model
        self.config = config

        self.train_loader = DataLoader(config.datasets['train'], batch_size=config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(config.datasets['valid'], batch_size=config.batch_size)
        self.test_loader = DataLoader(config.datasets['test'], batch_size=1)

        self.train_stats = []
        self.valid_stats = []


    def get_tqdm_desc(self, prefix: str) -> str:
        '''
        get tqdm description

        Args:
        ---
        - prefix

        Returns:
        ---
        - tqdm description
        '''
        out = prefix

        if self.current_epoch != None:
            out += f' {self.current_epoch + 1}'

        return out
    
    def step(
        self, 
        train_stats: TrainStatistics, 
        valid_stats: ValidationStatistics, 
        test_stats: TestStatistics
    ):
        '''
        step and collect statistics
        '''
        self.current_epoch += 1

        self.train_stats.append(train_stats.tuple())
        self.valid_stats.append(valid_stats.tuple())
    
    def state_dict(self) -> dict[str, any]:
        '''
        build state dict

        Returns:
        ---
        - state dict
        '''
        return {
            'model': self.model.state_dict(),
            'optim': self.config.optimizer.state_dict(),
            'lr_scheduler': self.config.lr_scheduler.state_dict(),
            'epoch': self.current_epoch,
            'train_stats': self.train_stats,
            'valid_stats': self.valid_stats
        }