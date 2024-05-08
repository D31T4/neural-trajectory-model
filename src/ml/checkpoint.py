import torch
import os
from pathlib import Path

DEFAULT_CHECKPOINT_DIR = f'{str(Path(__file__).parent)}/checkpoints'

class Checkpoint:
    '''
    checkpoint config
    '''
    
    def __init__(
        self, 
        checkpoint_interval: int, 
        prefix: str, 
        out_dir: str = DEFAULT_CHECKPOINT_DIR,
        should_save_best: bool = True
    ):
        '''
        Args:
        ---
        - checkpoint_interval: checkpoint every n epochs
        - prefix: save file prefix
        - out_dir: output directory
        '''
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.checkpoint_interval = checkpoint_interval
        self.prefix = prefix
        self.out_dir = out_dir
        self.should_save_best = should_save_best

    def should_checkpoint(self, epoch: int) -> bool:
        '''
        Args:
        ---
        - epoch: 0-based epoch
        '''
        return epoch > 0 and epoch % self.checkpoint_interval == 0
    
    def checkpoint(self, epoch: int, state_dict: dict[str, any]):
        '''
        checkpoint model
        '''
        torch.save(state_dict, f'{self.out_dir}/{self.prefix}_{epoch}.pt')

    def save_best(self, state_dict: dict[str, any]):
        '''
        save best model
        '''
        torch.save(state_dict, f'{self.out_dir}/{self.prefix}_best.pt')

    @staticmethod
    def none():
        return Checkpoint(float('inf'), 'placeholder')
