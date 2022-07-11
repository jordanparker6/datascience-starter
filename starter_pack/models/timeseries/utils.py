import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller

def check_stationary(y: np.ndarray, alpha: float = 0.05):
    result = adfuller(y)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return result[1] <= alpha

# check if leading or lagging
# https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
def plot_time_lagged_cross_corr(d1, d2):
    """
    Checks whether or not one series is leading or lagging the other.
    """
    def crosscorr(datax, datay, lag=0, wrap=False):
        """ Lag-N cross correlation. 
        Shifted data filled with NaNs 
        
        Parameters
        ----------
        lag : int, default 0
        datax, datay : pandas.Series objects of equal length
        Returns
        ----------
        crosscorr : float
        """
        if wrap:
            shiftedy = datay.shift(lag)
            shiftedy.iloc[:lag] = datay.iloc[-lag:].values
            return datax.corr(shiftedy)
        else: 
            return datax.corr(datay.shift(lag))

    seconds = 5
    fps = 30
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
    offset = np.floor(len(rs)/2)-np.argmax(np.abs(rs))
    f,ax=plt.subplots(figsize=(14,3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(np.abs(rs)),color='r',linestyle='--',label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', xlabel='Offset',ylabel='Pearson r')
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
    plt.legend()