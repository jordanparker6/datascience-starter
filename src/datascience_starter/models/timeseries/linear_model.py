import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline 
from sklearn.base import BaseEstimator
from datascience_starter.features.transformations import RBFFeatures


class LinearTimeseriesModel(BaseEstimator):
    """
    LinearTimeseriesModel
     A linear timeseries model leveraging exponentially decayed 
     weights to prioritise recent entries and RBF features to
     engineer seasonality
    
    args:
     -> alpha: hyperparameter for RBFFeatures
     -> r: hyperparameter for exponential decay
    """
    def __init__(self, alpha, r):
        super().__init__()
        self.params = { "alpha": alpha, "r": r }
        self.model = LinearRegression()

    def transform(self, df, ylabel='y', fit=False):
            rbf = RBFFeatures(self.params['alpha'])
            df = rbf.transform(df)
            X = df.drop([ylabel], axis=1)
            y = df[ylabel]
            return X, y

    def fit(self, df, ylabel='y'):
        X, y = self.transform(df, ylabel, fit=True)
        if self.params['r'] == 1:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y,  sample_weight=self._ewa(len(y), self.params['r']))
        return self.model

    def predict(self, df, ylabel='y'):
        X, y = self.transform(df, ylabel)
        pred = self.model.predict(X)
        return pred

    def score(self, df, ylabel='y', metric='mape'):
        X, y = self.transform(df, ylabel)
        if metric == 'mape':
            yhat = self.predict(df, ylabel)
            return self._mape(y, yhat)
        elif metric == 'r2':
            return self.model.score(X, y)
        else:
            return

    def _ewa(self, n, r):
        # Exponential Weighted Average
        # -> r [0,1]
        x = np.arange(0, n)
        func = lambda i: r ** i * (1 - r) / (1 - np.power(r, n))
        return np.flip(np.array([func(i) for i in x]))

    def _mape(self, y, yhat):
        return np.mean(np.abs((y - yhat) / y)) * 100