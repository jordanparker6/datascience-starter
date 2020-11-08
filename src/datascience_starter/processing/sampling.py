import numpy as np
from tqdm import tqdm

class Sampler:
    
    def bootstrap(self, X, n, stat=np.mean):
        for i in tqdm(range(n)):
            resample = np.random.choice(X, len(X))
            yield stat(X)

    def undersample(self, X, n):
        raise NotImplementedError

    def oversample(self, X, n):
        raise NotImplementedError

    