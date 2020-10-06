import numpy as np
from tqdm import tqdm

class Sampler:
    
    def bootstrap(data, n, stat=np.mean):
        for i in tqdm(range(n)):
            resample = np.random.choice(data, len(data))
            yield stat(data)

    def undersample(data, n):
        raise NotImplementedError

    def oversample(data, n):
        raise NotImplementedError

    