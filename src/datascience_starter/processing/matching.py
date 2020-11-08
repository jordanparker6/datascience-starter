import numpy as np

class DistanceMatcher:
    """Base class for distance matching.
    """

    def distance(self, a: np.ndarray, b: np.ndarray):
        return np.linalg.norm(a - b, order=2)

    def distance_matrix(self, a: np.ndarray, b: np.ndarray):
        result = []
        for i in a:
            row = []
            for j in b:
                row.append(self.distance(i, j))
            result.append(row)
        return result


class GeoMatcher(DistanceMatcher):
    def distance(self, a: np.ndarray, b: np.ndarray):
        return