import os
import pickle

from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_metrics(y, y_pred):

    roc = roc_auc_score(y, y_pred)

    ap = average_precision_score(y, y_pred)

    return roc, ap


def get_data_files():
    return [
        "satimage-2.mat",
        "satellite.mat",
        "pendigits.mat",
        "musk.mat",
    ]

def save_model(model, y_pred, experiment_dir, fold_idx, iter_idx):

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    experiment_name = f"fold{fold_idx}_iter{iter_idx}.pkl"

    with open(os.path.join(experiment_dir, experiment_name), "wb+") as f:
        pickle.dump((model, y_pred), f)


def save_scores(roc, ap, experiment_dir):

    with open(os.path.join(experiment_dir, "scores.csv"), "w+") as f:
        f.write(f"roc,ap\n{str(roc)},{str(ap)}")


def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
