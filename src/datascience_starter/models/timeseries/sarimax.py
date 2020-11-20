from sklearn.base import BaseEstimator
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAXModel
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any

class SARIMAX(BaseEstimator):
    """A class to fit and predict using the SARIMAX model.
     
     Implements the scikit-learn api for the SARIMAX model. 
     Also provides utility methods to check stationarity 
     with an Augmented Dickie Fuller test.
     
     Args:
       **kwargs: Key word arguements for the statsmodels SARIMAX class.
       Please see statsmodels for documentation.
    
     Attributes:
        params: The model paramaters.
        model: An instance of the statsmodels SARIMAX model.

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.params: Dict[str, Any] = kwargs
        self.model: SARIMAX = SARIMAX

    def fit(self, y: np.ndarray, max_iter: int = 50, method: str = 'powell', **kwargs):
        self.model = self.model(y, **self.params)
        self.model = self.model.fit(max_iter=max_iter, disp=0, method=method, **kwargs)
        return self

    def predict(self, X: np.ndarray):
        pred = self.model.get_prediction(start=X.index[0], end=X.index[-1])
        yhat = pred.predicted_mean
        ci = pred.conf_int()
        return yhat, ci

    def plot_predictions(self, X: np.ndarray, shift: int = 0):
        yhat, ci = self.predict(X)
        yhat = yhat[shift:]
        ci = ci.iloc[shift:, :]
        ax = X.plot(label='observed')
        yhat.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='b', alpha=.05)
        ax.set_xlabel('Date')
        ax.set_ylabel('y')
        plt.legend()
        plt.show()

    def forecast(self, start: str, end: str):
        pred = self.model.get_prediction(start=pd.to_datetime(start), end=pd.to_datetime(end))
        yhat = pred.predicted_mean
        ci = pred.conf_int()
        return yhat, ci

    def plot_forecast(self, start: str, end: str):
        yhat, ci = self.forecast(start, end)
        ax = yhat.plot(figsize=(14, 4), label="Forecast")
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='b', alpha=.05)
        ax.set_xlabel('Date')
        ax.set_ylabel('y')
        plt.legend()
        plt.show()

    def summary(self):
        print(self.model.summary().tables[1])
        self.model.plot_diagnostics(figsize=(18, 6))

    def score(self, y: np.ndarray, shift: int = 0):
        yhat, _ = self.predict(y)
        return self._mape(y[shift:].values, yhat[shift:].values)

    def check_stationary(self, y: np.ndarray, alpha: float = 0.05):
        result = adfuller(y)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        return result[1] <= alpha

    def _mape(self, y: np.ndarray, yhat: np.ndarray):
        return np.mean(np.abs((y - yhat) / y)) * 100