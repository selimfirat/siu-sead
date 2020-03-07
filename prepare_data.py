import pickle

import numpy as np
import os

from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import get_data_files
from pyod.utils.utility import standardizer

np.random.seed(1)


def stratified_cv(X, y, num_folds):

    folds = []
    skf = StratifiedKFold(n_splits=num_folds)

    splits = skf.split(X, y)

    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = standardizer(X_train, X_test)

        folds.append((X_train, y_train, X_test, y_test))

    return folds

def normal_instances(folds):

    new_folds = []
    for X_train, y_train, X_test, y_test in folds:
        new_fold = (X_train[np.argwhere(y_train==0).ravel(), :], y_train[np.argwhere(y_train == 0).ravel()], X_test, y_test)
        new_folds.append(new_fold)

    return new_folds


if __name__ == "__main__":

    num_folds = 5

    data_base_path = "data"
    data_files = get_data_files()

    data = {}

    for data_idx, data_file in tqdm(enumerate(data_files)):
        np.random.seed(1)

        data_path = os.path.join(data_base_path, data_file)

        data_name = data_file.split(".")[0]
        print("... Processing", data_name, '...')

        f = loadmat(data_path)

        X = f['X']
        y = f['y'].ravel()
        outliers_fraction = np.count_nonzero(y) / len(y)
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)

        scv_folds = stratified_cv(X, y, num_folds)
        ni_folds = normal_instances(scv_folds)

        data[data_name] = {
            "outliers_fraction": outliers_fraction,
            "stratified": scv_folds,
            "normals": ni_folds
        }

    target_path = os.path.join(data_base_path, "data.pkl")

    with open(target_path, "wb+") as f:
        pickle.dump(data, f)
