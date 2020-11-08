import itertools
import numpy as np
from typing import Dict, Any, Tupple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.base import BaseEstimator
from tqdm import tqdm

class GridsearchCVBase(BaseEstimator):
    """A base class for cross validated gridsearch.

    Args:
        estimator: A scikit learn stimator that implements the fit and score methods.
        cv: The number of folds in kfold cross validation.
    
    Attributes:
        estimator: A scikit learn stimator that implements the fit and score methods.
        cv: The number of folds in kfold cross validation.

    """
    def __init__(self, estimator, cv: int = 5):
        super().__init__()
        self.estimator = estimator
        self.cv = cv
        self.splitter = None

    def crossval(self, df: pd.DataFrame, parameters: Dict[str, Any], cv: int = 5) -> np.float:
        """Performs k-fold cross validation using the estimators score method and the provided splitter.

        Args:
            df: A pandas dataframe of target and feature variables.
            parameters: A dictionary of parameters and possible values.
            cv: The number of folds in k-fold cross validation.

        Returns:
            The mean score for the cross validation.

        """
        assert self.splitter != None, "No splitter specified"

        cv = self.splitter(n_splits=cv)
        score = []
        for train_index, test_index in cv.split(df):
            train, test = df.iloc[train_index, :], df.iloc[test_index, :]
            model = self.estimator(**parameters)
            model.fit(train)
            score.append(model.score(test))
        return np.array(score).mean()

    def fit(self, df: pd.DataFrame, parameters: Dict[str, Any], min_loss: bool = True) -> Tupple(Dict[str, Any], np.ndarray):
        """Fit method for cross validated grid search.

        Args:
            df: A pandas dataframe of target and feature variables.
            parameters: A dictionary of parameters and possible values.
            min_loss: A boolean indicator to optimise for the min or max score in gridsearch.

        """
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
    """"A gridsearch and crossvalidation approach for iid datasets.
    """
    def __init__(self, estimator, cv: int = 5):
        super().__init__()
        self.splitter = ShuffleSplit


class TimeseriesGridsearchCV(GridsearchCVBase):
    """" A gridsearch and crossvalidation approach for timeseries datasets.
    """
    def __init__(self, estimator, cv=5):
        super().__init__()
        self.splitter = TimeSeriesSplit