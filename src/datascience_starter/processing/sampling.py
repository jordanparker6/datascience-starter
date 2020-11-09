import numpy as np
from typing import Callable
from tqdm import tqdm

class Sampler:
    """A class to provide sampling methods.
    """
    
    def bootstrap(self, X: np.ndarray, n: int, stat: Callable = np.mean):
        """A method to implement a bootstrap.

        Bootstrapping is a sampling technique to develop an estimate of 
        the distribution of a sample statistic. A bootstrap samples with
        replacement from the original sample for len(X) samples and calculates
        the statistic over the new sample. This process is repeated 'n' times.

        Args:
            X: A 1-D array of features to bootstrap.
            n: The number of bootstraps to sample.
            stat (optional): The statistic to compute on the bootstrap. Defaults to np.mean.

        Yields:
            np.float: The statistic on the bootstrapped sample.
        """
        for i in tqdm(range(n)):
            resample = np.random.choice(X, len(X))
            yield stat(X)

    def undersample(self, X, n):
        raise NotImplementedError

    def oversample(self, X, n):
        raise NotImplementedError

    