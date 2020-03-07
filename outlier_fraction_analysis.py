import os

import numpy as np
from tqdm import tqdm
from utils import calculate_metrics
from best_models import cfg, all_data, models_dict, num_experiments

for data_name, data in tqdm(all_data.items()):
        fname = f"figures/outlier_fraction_{data_name}.csv"

        if os.path.exists(fname):
            continue

        res = "model,data,fraction,roc,ap\n"

        for model_name, model in tqdm(models_dict.items()):
            model_cls = models_dict[model_name]["cls"]
            params = models_dict[model_name]["params"]

            for fraction in tqdm(np.arange(0.01, 1.01, 0.01)):
                experiment_type = "stratified"
                experiment_dir = f"experiments/{experiment_type}_{data_name}_{model_name}"

                roc = 0.0
                ap = 0.0

                folds = data[experiment_type]

                for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
                    np.random.seed(1)
                    indices = np.concatenate([np.argwhere(y_train == 0).ravel(), np.random.choice(np.argwhere(y_train == 1).ravel(), int(
                        fraction * len(np.argwhere(y_train == 1).ravel())), replace=False)])
                    X_train_T = X_train[indices, :]

                    for iter_idx in range(cfg["num_iterations"]):
                        np.random.seed(iter_idx)
                        model = model_cls(**params)

                        model.fit(X_train_T)

                        y_test_pred = model.predict_proba(X_test)[:, 1]

                        iter_roc, iter_ap = calculate_metrics(y_test, y_test_pred)

                        roc += iter_roc
                        ap += iter_ap

                roc = np.round(roc / num_experiments, decimals=4)
                ap = np.round(ap / num_experiments, decimals=4)

                res += f"{model_name},{data_name},{str(fraction)},{str(roc)},{str(ap)}\n"

        res_f = open(fname, "w+")

        res_f.write(res)
