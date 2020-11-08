from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class RBFFeatures(BaseEstimator, TransformerMixin):
    """
    RBFFeatures
     Builds a set of monthly Radial Basis Function features from a
     pandas datetime index.

     args:
      -> alpha: a smoothing hyperparamter to controll hump widths
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.months = ['jan', 'feb', 'mar', 'apr', 'may','jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    def rbf(self, t, month, alpha):
        # Radial Basis Function
        # t --> time in month units
        # month --> position of peak
        # alpha --> width tunning parameter
        t = t % 12
        return np.exp(-1 / (2 * alpha) * np.power(t - month, 2))
        
    def transform(self, X):
        data = { k: self.rbf(X.index.month, i, self.alpha) for i, k in enumerate(self.months) }
        data = pd.DataFrame(data, index=X.index)
        return pd.concat([data, X], axis=1)