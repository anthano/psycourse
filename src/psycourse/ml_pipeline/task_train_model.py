import pickle

import matplotlib.pyplot as plt
import pandas as pd

from psycourse.config import BLD_DATA, BLD_MODELS, BLD_RESULTS, SRC
from psycourse.ml_pipeline.train_model import svm_model

task_products = {
    "probabilities": BLD_DATA / "svm_predicted_probabilities.pkl",
    "model": BLD_MODELS / "svm_classifier.pkl",
    "learning_curve": BLD_RESULTS / "plots/svm/learning_curve.svg",
    "roc_curves": BLD_RESULTS / "plots/svm/roc_curves.svg",
}


def task_train_model(
    script_path=SRC / "ml_pipeline" / "train_model.py",
    data_path=BLD_DATA / "sparse_dataset_with_targets.pkl",
    produces=task_products,
):
    """Train SVM Model and save the predicted probabilities."""

    data = pd.read_pickle(data_path)
    final_model, cluster_probability_scores_df, learning_curve_fig, roc_fig = svm_model(
        data
    )
    cluster_probability_scores_df.to_pickle(task_products["probabilities"])

    with open(task_products["model"], "wb") as f:
        pickle.dump(final_model, f)

    learning_curve_fig.savefig(task_products["learning_curve"], bbox_inches="tight")
    plt.close(learning_curve_fig)

    roc_fig.savefig(task_products["roc_curves"], bbox_inches="tight")
    plt.close(roc_fig)
