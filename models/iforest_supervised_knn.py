# -*- coding: utf-8 -*-

from pyod.models.base import BaseDetector
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
import numpy as np


class IForestSupervisedKNN(BaseDetector):

    def __init__(self, get_top=0.8, if_params={}, knn_params={}):
        super(IForestSupervisedKNN, self).__init__()
        self.get_top = get_top
        self.is_fitted = False

        self.iforest = IForest(**if_params)

        self.knn = KNN(**knn_params)

    def fit(self, X, y=None):

        X = check_array(X)
        self._set_n_classes(y)

        self.iforest.fit(X)

        scores = self.iforest.predict_proba(X)[:, 1]

        normal_instances = X[np.argsort(scores)[:int(len(X)*self.get_top)]]

        self.knn.fit(normal_instances)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        self.is_fitted = True

        return self

    def decision_function(self, X):

        check_is_fitted(self, ['is_fitted'])

        return self.knn.decision_function(X)
