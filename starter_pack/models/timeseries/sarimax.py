from sklearn.base import BaseEstimator
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAXModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm
from .utils import check_stationary

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
        self.model: SARIMAXModel = SARIMAXModel

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
        return check_stationary(y, alpha)

    def _mape(self, y: np.ndarray, yhat: np.ndarray):
        return np.mean(np.abs((y - yhat) / y)) * 100

def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    
    results = []
    
    for param in tqdm(parameters_list):
        try: 
            model = SARIMAX(
                exog, 
                order=(param[0], d, param[1]), 
                seasonal_order=(param[2], D, param[3], s)
            ).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([param, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df