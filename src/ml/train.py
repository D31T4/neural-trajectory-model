'''
Implements training pipeline
'''

import torch
import torch.nn.functional as F

import tqdm
import time

from src.ml.model import TrajectoryModel
from src.ml.config import TrainConfig
from src.ml.stats import TrainStatistics, ValidationStatistics, TestStatistics
from src.ml.state import TrainState
from src.ml.utils import accuracy, mean_deviation


def train_one_epoch(state: TrainState) -> TrainStatistics:
    '''
    train one epoch using training set

    Args:
    ---
    - state: train state
    '''
    model = state.model
    config = state.config
    
    model.train()

    train_acc = 0
    train_loss = 0
    train_nll = 0
    train_count = 0

    all_candidates = config.all_candidates

    if config.preprocess:
        all_candidates = config.preprocess(all_candidates)

    start_time = time.time()

    for trajectories, context, target in tqdm.tqdm(state.train_loader, desc=state.get_tqdm_desc('[train]'), disable=not config.verbose):
        batch_size, sequence_length = trajectories.shape[:2]
        
        context: torch.FloatTensor = context # type hint hack
        trajectories: torch.FloatTensor = trajectories
        target: torch.IntTensor = target


        if config.cuda:
            context = context.cuda()
            trajectories = trajectories.cuda()
            target = target.cuda()

        if config.preprocess:
            trajectories = config.preprocess(trajectories)
        
        logits: torch.Tensor = model(
            context[:, :-1], 
            trajectories[:, :-1], 
            all_candidates,
            context
        )

        # parametric crosss-entropy loss same as language modelling
        loss = F.cross_entropy(
            logits[:, 1:].reshape(-1, all_candidates.shape[0]),
            target[:, 1:].reshape(-1)
        )

        # throw if nan loss
        assert not torch.isnan(loss).item()

        config.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        config.optimizer.step()

        # accumulate training loss
        train_loss += loss.item()

        # accumulate accuracy
        acc = accuracy(logits, target).item()
        train_acc += acc

        train_count += batch_size

    end_time = time.time()

    return TrainStatistics(
        train_loss / len(state.train_loader), 
        train_loss / len(state.train_loader), 
        train_acc / len(state.train_loader), 
        train_count,
        elapsed=end_time - start_time
    )

def validate_one_epoch(state: TrainState) -> ValidationStatistics:
    '''
    evaluate model using validation set

    Args:
    ---
    - state: train state
    '''
    model = state.model
    config = state.config

    model.eval()

    valid_acc = 0
    valid_nll = 0
    valid_mse = 0
    valid_count = 0

    all_candidates = config.all_candidates

    if config.preprocess:
        all_candidates = config.preprocess(all_candidates)

    start_time = time.time()
    
    with torch.no_grad():
        for trajectories, context, target in tqdm.tqdm(state.valid_loader, desc=state.get_tqdm_desc('[valid]'), disable=not config.verbose):
            batch_size, sequence_length = trajectories.shape[:2]
        
            context: torch.FloatTensor = context # type hint hack
            trajectories: torch.FloatTensor = trajectories
            target: torch.IntTensor = target

            if config.cuda:
                context = context.cuda()
                trajectories = trajectories.cuda()
                target = target.cuda()

            if config.preprocess:
                trajectories = config.preprocess(trajectories)

            logits: torch.Tensor = model(
                context[:, :-1], 
                trajectories[:, :-1], 
                all_candidates,
                context
            )

            # accumulate nll
            nll = F.cross_entropy(
                logits[:, 1:].reshape(-1, all_candidates.shape[0]),
                target[:, 1:].reshape(-1)
            ).item()
            
            valid_nll += nll
            
            # accumulate accuracy
            acc = accuracy(logits, target).item()
            valid_acc += acc

            # accumulate mse
            mdev = mean_deviation(logits, target, config.all_candidates).item()
            valid_mse += mdev

            valid_count += batch_size

    end_time = time.time()

    return ValidationStatistics(
        nll=valid_nll / len(state.valid_loader), 
        acc=valid_acc / len(state.valid_loader), 
        sample_size=valid_count,
        mdev=valid_mse / len(state.valid_loader),
        elapsed=end_time - start_time
    )

def test_one_epoch(state: TrainState) -> TestStatistics:
    '''
    test model using test set. Currently a placeholder.

    Args:
    ---
    - state: train state
    '''
    model = state.model
    config = state.config

    model.eval()

    all_candidates = config.all_candidates

    if config.preprocess:
        all_candidates = config.preprocess(all_candidates)

    start_time = time.time()

    with torch.no_grad():
        for trajectories, context, target in tqdm.tqdm(state.test_loader, desc=state.get_tqdm_desc('[test]'), disable=not config.verbose):
            pass

    end_time = time.time()

    return TestStatistics(
        0, 
        0, 
        0,
        elapsed=end_time - start_time
    )

def train(model: TrajectoryModel, config: TrainConfig):
    '''
    train model.

    Args:
    ---
    - model: model to be trained
    - config: train config
    '''
    state = TrainState(model, config)

    # min validation perplexity
    min_valid_perplexity: float = float('inf')

    for _ in range(config.n_epoch):
        train_stats: TrainStatistics = train_one_epoch(state)
        if config.verbose: train_stats.report()
        
        valid_stats: ValidationStatistics = validate_one_epoch(state)
        if config.verbose: valid_stats.report()

        test_stats: TestStatistics = test_one_epoch(state)
        if config.verbose: test_stats.report()

        # checkpoint model
        if config.checkpoint.should_checkpoint(state.current_epoch):
            config.checkpoint.checkpoint(state.current_epoch, state.state_dict())

        # save best model
        if valid_stats.nll < min_valid_perplexity and config.checkpoint.should_save_best:
            config.checkpoint.save_best(state.state_dict())
            min_valid_perplexity = valid_stats.nll

        state.step(train_stats, valid_stats, test_stats)
        config.lr_scheduler.step()

    return state
