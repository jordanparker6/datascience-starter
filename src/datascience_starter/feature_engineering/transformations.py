import numpy as np
import pandas as pd
import sklearn.base as base

class RBFFeatures(base.TransformerMixin):
    """Builds a set of monthly Radial Basis Function features from a pandas datetime index.

     Args:
        alpha: A smoothing hyperparamter between (0,1) to control hump widths.

    """
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha  #: A smoothing hyperparamter between (0,1) to control hump widths.
        self.months = ['jan', 'feb', 'mar', 'apr', 'may','jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']   #: An array of month strings.

    def _rbf(self, t: float, month: int, alpha: float) -> np.ndarray:
        """Radial basis function

        Args:
            t: Continuous time in month units.
            month: The position of the peak.
            alpha: Width tuning parameter between (0, 1).
        
        Returns:
            A numpy array of radial basis tranformed values.

        """
        t = t % 12
        return np.exp(-1 / (2 * alpha) * np.power(t - month, 2))
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Transforms a pandas datetime index to RBF Features.

        Args:
            X: A pandas dataframe with a datetime index.
        
        Returns:
            A pandas dataframe with RBF Features concatenated to the dataframe.
        """
        data = { k: self._rbf(X.index.month, i, self.alpha) for i, k in enumerate(self.months) }
        data = pd.DataFrame(data, index=X.index)
        return pd.concat([data, X], axis=1)