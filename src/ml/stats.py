import math

class Statistics:
    def __init__(
        self,
        nll: float,
        acc: float,
        sample_size: int,
        elapsed: float = -1
    ):
        '''
        Args:
        ---
        - nll: negative log-likelihood
        - acc: accuracy
        - sample_size: sample size
        - elapsed: elapsed time
        '''
        self.nll = nll
        self.acc = acc
        self.sample_size = sample_size
        self.elapsed = elapsed

    def report(self):
        '''
        report statistics
        '''
        if self.elapsed > 0:
            print(f'elapsed: {self.elapsed}')

        print(f'perplexity: {math.exp(self.nll)}')
        print(f'accuracy: {self.acc}')

    def tuple(self):
        '''
        convert to tuple
        '''
        return (self.nll,)

class TrainStatistics(Statistics):
    '''
    train statistics
    '''

    def __init__(
        self,
        avg_loss: float,
        nll: float,
        acc: float,
        sample_size: int,
        elapsed: float = 0
    ):
        '''
        Args:
        ---
        - avg_loss: average loss
        - nll: negative log-likelihood
        - acc: accuracy
        - sample_size: sample size
        - elapsed: elapsed time
        '''
        super().__init__(nll, acc, sample_size, elapsed)
        
        self.avg_loss = avg_loss

    def report(self):
        print(f'loss: {self.avg_loss}')
        super().report()

    def tuple(self):
        return self.nll, self.avg_loss, self.acc

class ValidationStatistics(Statistics):
    '''
    validation statistics
    '''
    def __init__(
        self,
        nll: float,
        acc: float,
        mdev: float,
        sample_size: int,
        elapsed: float = -1
    ):
        '''
        Args:
        ---
        - nll: negative log-likelihood
        - acc: accuracy
        - mdev: mean error
        - sample_size: sample size
        - elapsed: elapsed time
        '''
        super().__init__(nll, acc, sample_size, elapsed)

        self.mdev = mdev

    def report(self):
        print(f'mdev: {self.mdev}')
        super().report()

    def tuple(self):
        return self.nll, self.mdev, self.acc

class TestStatistics(Statistics):
    '''
    test statistics
    '''
    pass
