import numpy as np
from enum import Enum
from typing import Callable
from tqdm import tqdm

class BootstrapMethods(Enum):
    ORDINARY = 'ordinary'
    BALANCED = 'balanced'

class Sampler:
    """A class to provide sampling methods.
    """
    
    def bootstrap(self, X: np.ndarray, size: int, stat: Callable = np.mean, method: BootstrapMethods = 'balanced'):
        """A method to implement a bootstrap.

        Bootstrapping is a sampling technique to develop an estimate of 
        the distribution of a sample statistic. A bootstrap samples with
        replacement from the original sample for N samples and calculates
        the statistic over the new sample. This process is repeated 'n' times.

        Two sampling methods are implemented. An ordinary sample is a simple
        sample with replacement and balanced utlises latin hypercube sampling
        to sample from all n bins in the statespace.

        Args:
            X: A 1-D array of features to bootstrap.
            size: The number of bootstraps to sample.
            stat (optional): The statistic to compute on the bootstrap. Defaults to np.mean.
            method (optional): The sampling method of the bootstrap

        Yields:
            np.float: The statistic on the bootstrapped sample.
        """
        n = len(X)
        idx = np.random.permutation(n * size)
        for i in tqdm(range(size)):
            if method == 'balanced':
                sel = idx[i * n : (i + 1) * n] % n
                yield stat(X[sel])
            else:
                resample = np.random.choice(X, n, replace=True)
                yield stat(resample)

    def undersample(self, X, n):
        raise NotImplementedError

    def oversample(self, X, n):
        raise NotImplementedError

    