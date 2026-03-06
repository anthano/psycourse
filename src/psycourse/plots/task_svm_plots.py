import pickle

import matplotlib.pyplot as plt

from psycourse.config import BLD_RESULTS, SRC
from psycourse.plots.svm_plots import plot_learning_curve, plot_roc_curves

_PLOT_DIR = BLD_RESULTS / "plots" / "svm"

task_products = {
    "learning_curve": _PLOT_DIR / "learning_curve.svg",
    "roc_curves": _PLOT_DIR / "roc_curves.svg",
}


def task_plot_svm(
    script_path=SRC / "plots" / "svm_plots.py",
    learning_curve_data_path=BLD_RESULTS / "svm" / "learning_curve_data.pkl",
    roc_data_path=BLD_RESULTS / "svm" / "roc_data.pkl",
    produces=task_products,
):
    """Render and save SVM diagnostic plots from pre-computed data.

    This task is independent of task_train_model: re-running it only requires
    the serialised data dicts, not a full model refit.
    """
    with open(learning_curve_data_path, "rb") as f:
        learning_curve_data = pickle.load(f)

    with open(roc_data_path, "rb") as f:
        roc_data = pickle.load(f)

    fig = plot_learning_curve(learning_curve_data)
    fig.savefig(task_products["learning_curve"], bbox_inches="tight")
    plt.close(fig)

    fig = plot_roc_curves(roc_data)
    fig.savefig(task_products["roc_curves"], bbox_inches="tight")
    plt.close(fig)
