import os
import pickle

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

res = "Dataset,num_samples,num_dims,outliers_fraction\n"

with open("data/data.pkl", "rb") as f:
    all_data = pickle.load(f)
    for data_name, data_dict in all_data.items():
        X_train, y_train, X_test, y_test = data_dict["stratified"][0]

        outliers = data_dict["outliers_fraction"]
        dims = X_train.shape[1]
        num_samples = len(X_train) + len(X_test)
        res += f"{data_name},{num_samples},{dims},{outliers}\n"


with open("figures/datasets.csv", "w+") as f:
    f.write(res)
