import itertools
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator
from tqdm import tqdm

class TimeseriesGridsearchCV(BaseEstimator):
    """"
    TimeseriesGirdsearchCV
     A combined gridsearch and crossvalidation approach for timeseries datasets.
     Cross Validation splits the dataset into k folds where the train is an expanding set of history
     and the train is an equal length hold out period.
    Args:
       -> estimater: a scikit learn estimator with a fit and score method
       -> cv: the number of folds in kfold cross validation
    """
    def __init__(self, estimator, cv=5):
        super().__init__()
        self.estimator = estimator
        self.cv = cv

    def _crossval(self, df, parameters, cv=5):
        tscv = TimeSeriesSplit(n_splits=cv)
        score = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index, :], df.iloc[test_index, :]
            model = self.estimator(**parameters)
            model.fit(train)
            score.append(model.score(test))
        return np.array(score).mean()

    def fit(self, df, parameters, min_loss=True):
        scores = []
        params = []
        values = parameters.values()
        options = [dict(zip(parameters.keys(), v)) for v in itertools.product(*parameters.values())]
        for option in tqdm(options):
            score = self._crossval(df, option, self.cv)
            scores.append(score)
            params.append(option)
        scores = np.array(scores)
        if min_loss:
            best = np.nanargmin(scores)
        else:
            best = np.nanargmax(scores)
        return params[best], scores