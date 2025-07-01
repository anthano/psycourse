import pickle

import pandas as pd

from psycourse.config import BLD_DATA, BLD_MODELS, SRC
from psycourse.ml_pipeline.train_model import svm_model

task_products = {
    "probabilities": BLD_DATA / "svm_predicted_probabilities.pkl",
    "model": BLD_MODELS / "svm_classifier.pkl",
}


def task_train_model(
    script_path=SRC / "ml_pipeline" / "train_model.py",
    data_path=BLD_DATA / "sparse_dataset_with_targets.pkl",
    produces=task_products,
):
    """Train SVM Model and save the predicted probabilities."""

    data = pd.read_pickle(data_path)
    final_model, cluster_probability_scores_df = svm_model(data)
    cluster_probability_scores_df.to_pickle(task_products["probabilities"])

    with open(task_products["model"], "wb") as f:
        pickle.dump(final_model, f)
