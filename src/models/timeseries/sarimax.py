from sklearn.base import BaseEstimator
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAXModel
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SARIMAX(BaseEstimator):
    """
    SARIMAX
     Provides a class to fit and predict using the SARIMAX model.
     Implements the scikit-learn api.
     
     Args:
       -> Please see statsmodels for list of SARIMAX arguments.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs

    def fit(self, y, max_iter=50, method='powell', **kwargs):
        self.model = SARIMAXModel(y, **self.params)
        self.model = self.model.fit(max_iter=max_iter, disp=0, method=method, **kwargs)
        return self

    def predict(self, X):
        pred = self.model.get_prediction(start=X.index[0], end=X.index[-1])
        yhat = pred.predicted_mean
        ci = pred.conf_int()
        return yhat, ci

    def plot_predictions(self, X, shift=0):
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

    def forecast(self, start, end):
        pred = self.model.get_prediction(start=pd.to_datetime(start), end=pd.to_datetime(end))
        yhat = pred.predicted_mean
        ci = pred.conf_int()
        return yhat, ci

    def plot_forecast(self, start, end):
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

    def score(self, y, shift=0):
        yhat, _ = self.predict(y)
        return self._mape(y[shift:].values, yhat[shift:].values)

    def check_stationary(self, y, alpha=0.05):
        result = adfuller(y)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        return result[1] <= alpha

    def _mape(self, y, yhat):
        return np.mean(np.abs((y - yhat) / y)) * 100