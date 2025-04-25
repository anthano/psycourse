import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.ml_pipeline.train_model import svm_model


def task_train_model(
    script_path=SRC / "ml_pipeline" / "train_model.py",
    data_path=BLD_DATA / "sparse_dataset_with_targets.pkl",
    produces=BLD_DATA / "svm_predicted_probabilities.pkl",
):
    """Train SVM Model and save the predicted probabilities."""

    data = pd.read_pickle(data_path)
    cluster_probability_scores_df = svm_model(data)
    cluster_probability_scores_df.to_pickle(produces)
