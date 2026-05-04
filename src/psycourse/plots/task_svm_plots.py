import pickle

import matplotlib.pyplot as plt

from psycourse.config import BLD_RESULTS, SRC
from psycourse.plots.svm_plots import (
    plot_confusion_matrix,
    plot_learning_curve,
    plot_nested_cv_scores,
    plot_per_class_metrics,
    plot_roc_curves,
    plot_svm_combined,
)

_PLOT_DIR = BLD_RESULTS / "plots" / "svm"
_SVM_DIR = BLD_RESULTS / "svm"

task_products = {
    "learning_curve": _PLOT_DIR / "learning_curve.svg",
    "roc_curves": _PLOT_DIR / "roc_curves.svg",
    "confusion_matrix": _PLOT_DIR / "confusion_matrix.svg",
    "per_class_metrics": _PLOT_DIR / "per_class_metrics.svg",
    "nested_cv_scores": _PLOT_DIR / "nested_cv_scores.svg",
    "combined": _PLOT_DIR / "svm_combined.svg",
}


def task_plot_svm(
    script_path=SRC / "plots" / "svm_plots.py",
    learning_curve_data_path=_SVM_DIR / "learning_curve_data.pkl",
    roc_data_path=_SVM_DIR / "roc_data.pkl",
    nested_scores_data_path=_SVM_DIR / "nested_scores_data.pkl",
    cm_data_path=_SVM_DIR / "cm_data.pkl",
    report_dict_path=_SVM_DIR / "report_dict.pkl",
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
    with open(nested_scores_data_path, "rb") as f:
        nested_scores_data = pickle.load(f)
    with open(cm_data_path, "rb") as f:
        cm_data = pickle.load(f)
    with open(report_dict_path, "rb") as f:
        report_dict = pickle.load(f)

    # ── Individual plots ──────────────────────────────────────────────────────
    fig = plot_learning_curve(learning_curve_data)
    fig.savefig(task_products["learning_curve"], bbox_inches="tight")
    fig.savefig(
        task_products["learning_curve"].with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plot_roc_curves(roc_data)
    fig.savefig(task_products["roc_curves"], bbox_inches="tight")
    fig.savefig(
        task_products["roc_curves"].with_suffix(".tiff"), dpi=600, bbox_inches="tight"
    )
    plt.close(fig)

    fig = plot_confusion_matrix(cm_data)
    fig.savefig(task_products["confusion_matrix"], bbox_inches="tight")
    fig.savefig(
        task_products["confusion_matrix"].with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plot_per_class_metrics(report_dict)
    fig.savefig(task_products["per_class_metrics"], bbox_inches="tight")
    fig.savefig(
        task_products["per_class_metrics"].with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plot_nested_cv_scores(nested_scores_data)
    fig.savefig(task_products["nested_cv_scores"], bbox_inches="tight")
    fig.savefig(
        task_products["nested_cv_scores"].with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ── Combined 2 × 2 figure ─────────────────────────────────────────────────
    fig = plot_svm_combined(
        learning_curve_data=learning_curve_data,
        roc_data=roc_data,
        cm_data=cm_data,
        report_dict=report_dict,
    )
    fig.savefig(task_products["combined"], bbox_inches="tight")
    fig.savefig(
        task_products["combined"].with_suffix(".tiff"), dpi=600, bbox_inches="tight"
    )
    plt.close(fig)
