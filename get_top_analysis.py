import os

import numpy as np
from tqdm import tqdm
from utils import calculate_metrics
from best_models import cfg, all_data, models_dict, num_experiments


model_name = "IFSKNN"
model_cls = models_dict[model_name]["cls"]
params = models_dict[model_name]["params"]
fname = f"figures/get_top_parameter.csv"

if os.path.exists(fname):
    exit(0)

res = "model,data,get_top,roc,ap\n"

for data_name, data in tqdm(all_data.items()):

    for get_top in tqdm(np.arange(0.01, 1.01, 0.01)):
        experiment_type = "stratified"
        experiment_dir = f"experiments/{experiment_type}_{data_name}_{model_name}"

        roc = 0.0
        ap = 0.0

        folds = data[experiment_type]

        for fold_idx, (X_train, _, X_test, y_test) in enumerate(folds):

            for iter_idx in range(cfg["num_iterations"]):
                np.random.seed(iter_idx)

                params.update({ "get_top": get_top })
                model = model_cls(**params)

                model.fit(X_train)

                y_test_pred = model.predict_proba(X_test)[:, 1]

                iter_roc, iter_ap = calculate_metrics(y_test, y_test_pred)

                roc += iter_roc
                ap += iter_ap

        roc = np.round(roc / num_experiments, decimals=4)
        ap = np.round(ap / num_experiments, decimals=4)

        res += f"{model_name},{data_name},{str(get_top)},{str(roc)},{str(ap)}\n"

    res_f = open(fname, "w+")

    res_f.write(res)
