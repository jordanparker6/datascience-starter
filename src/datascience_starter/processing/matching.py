import numpy as np

class DistanceMatcher:
    """Base class for distance matching.
    """

    def distance(self, x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y, order=2)

    def distance_matrix(self, x: np.ndarray, y: np.ndarray):
        result = []
        for i in x:
            for j in y:
                pass
