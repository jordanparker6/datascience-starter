import itertools
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import SuffleSplit
from sklearn.base import BaseEstimator
from tqdm import tqdm

class GridsearchCVBase(BaseEstimator):
    """
    GirdsearchCVBase
     A base class for cross validated gridsearch.
    Args:
       -> estimater: a scikit learn estimator with a fit and score method
       -> cv: the number of folds in kfold cross validation
    """
    def __init__(self, estimator, cv=5):
        super().__init__()
        self.estimator = estimator
        self.cv = cv

    def crossval(self, df, parameters, cv=5):
        raise NotImplementedError

    def fit(self, df, parameters, min_loss=True):
        scores = []
        params = []
        values = parameters.values()
        options = [dict(zip(parameters.keys(), v)) for v in itertools.product(*parameters.values())]
        for option in tqdm(options):
            score = self.crossval(df, option, self.cv)
            scores.append(score)
            params.append(option)
        scores = np.array(scores)
        if min_loss:
            best = np.nanargmin(scores)
        else:
            best = np.nanargmax(scores)
        return params[best], scores


class GridsearchCV(GridsearchCVBase):
    """"
    GirdsearchCV
     A combined gridsearch and crossvalidation approach for iid datasets.
     Uses shuffle split to randomly generate k-folds.
    """
    def __init__(self, estimator, cv=5):
        super().__init__()

    def crossval(self, df, parameters, cv=5):
        cv = SuffleSplit(n_splits=cv)
        score = []
        for train_index, test_index in cv.split(df):
            train, test = df.iloc[train_index, :], df.iloc[test_index, :]
            model = self.estimator(**parameters)
            model.fit(train)
            score.append(model.score(test))
        return np.array(score).mean()

class TimeseriesGridsearchCV(GridsearchCVBase):
    """"
    TimeseriesGirdsearchCV
     A combined gridsearch and crossvalidation approach for timeseries datasets.
     Uses time series split to split k-folds based on a expanding set of history.
    """
    def __init__(self, estimator, cv=5):
        super().__init__()

    def crossval(self, df, parameters, cv=5):
        tscv = TimeSeriesSplit(n_splits=cv)
        score = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index, :], df.iloc[test_index, :]
            model = self.estimator(**parameters)
            model.fit(train)
            score.append(model.score(test))
        return np.array(score).mean()