import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ScoreScaler(BaseEstimator, TransformerMixin):
    """Transforms features by scaling each feature to given scoring scale.

    This estimator scales and translates each feature individually such
    that it acccords with a given range on the training set, e.g. between
    zero and one. Without scale arguments, ScoreScaler acts like MinMaxScaler.

    Parameters
    ----------
    scores_old_min : int, float, or 'auto'; default 'auto'
        The smallest/worst score on the original scale. If 'auto', the smallest value of
        each feature is assumed to be the smallest possible value.

    scores_old_max : int, float, or 'auto'; default 'auto'
        The highest/best score on the original scale. If 'auto', the greatest value of
        each feature is assumed to be the highest possible value.

    scores_new_min : int or float; default 0
        The smallest/worst score on the transformed scale.

    scores_new_max : int or float; default 1
        The highest/best score on the transformed scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    """

    def __init__(self, scores_old_min='auto', scores_old_max='auto', scores_new_min=0, scores_new_max=1):
        self.scores_old_min = scores_old_min
        self.scores_old_max = scores_old_max
        self.scores_new_min = scores_new_min
        self.scores_new_max = scores_new_max

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling, if no score range is given.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """

        if self.scores_old_min == 'auto':
            self.scores_old_min_ = X.min()
        else:
            self.scores_old_min_ = self.scores_old_min

        if self.scores_old_max == 'auto':
            self.scores_old_max_ = X.max()
        else:
            self.scores_old_max_ = self.scores_old_max
        return self

    def transform(self, X):
        """Scaling features of X according to scale settings.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """

        X = self.scores_new_max - ((self.scores_new_max - self.scores_new_min) *
                                   (self.scores_old_max_ - X) / (self.scores_old_max_ - self.scores_old_min_))
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to scale settings.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """

        X = self.scores_old_max_ - ((self.scores_old_max_ - self.scores_old_min_) *
                                   (self.scores_new_max - X) / (self.scores_new_max - self.scores_new_min))
        return X


