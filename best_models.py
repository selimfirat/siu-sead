import argparse
import os
import pickle
import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM

from models.iforest_supervised_knn import IForestSupervisedKNN
from utils import calculate_metrics, save_model, save_scores
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Sequential Ensembling of IForest and KNN')
parser.add_argument("--random_state", default=1, type=int)
parser.add_argument("--train_on", default="stratified_cv", type=str)
parser.add_argument('--num_folds', default=5, type=int)
parser.add_argument("--num_iterations", default=3, type=int)
cfg = vars(parser.parse_args())

data_file = "data/data.pkl"

if not os.path.exists(data_file):
    raise Exception("Run 'python prepare_data.py' first.")


num_experiments = cfg["num_iterations"] * cfg["num_folds"]

all_data = pickle.load(open(data_file, "rb"))

if_params = {"behaviour": "new"}
knn_params = { }


models_dict = {
    "IF": {
        "cls": IForest,
        "params": if_params
    },
    "KNN": {
        "cls": KNN,
        "params": knn_params
    },
    "IFSKNN": {
        "cls": IForestSupervisedKNN,
        "params": { "knn_params": knn_params, "if_params": if_params, "get_top": 0.9 }
    },
    "OCSVM": {
        "cls": OCSVM,
        "params": { }
    }
}

if __name__ == "__main__":
    for data_name, data in tqdm(all_data.items()):
        for model_name, model_dict in tqdm(models_dict.items()):
            for experiment_type in ["normal_instances", "stratified_cv"]:

                experiment_dir = f"experiments/{experiment_type}_{data_name}_{model_name}"

                if os.path.exists(experiment_dir):
                    continue


                roc = 0.0
                ap = 0.0

                if experiment_type not in data:
                    experiment_type = experiment_type.replace("_cv", "").replace("_instances", "")

                folds = data[experiment_type]

                for fold_idx, (X_train, _, X_test, y_test) in enumerate(folds):

                    for iter_idx in range(cfg["num_iterations"]):
                        np.random.seed(iter_idx)

                        model = model_dict["cls"](**model_dict["params"])

                        model.fit(X_train)

                        y_test_pred = model.predict_proba(X_test)[:, 1]

                        iter_roc, iter_ap = calculate_metrics(y_test, y_test_pred)

                        save_model(model, y_test_pred, experiment_dir, fold_idx, iter_idx)

                        roc += iter_roc
                        ap += iter_ap

                roc = np.round(roc / num_experiments, decimals=4)
                ap = np.round(ap / num_experiments, decimals=4)

                save_scores(roc, ap, experiment_dir)
