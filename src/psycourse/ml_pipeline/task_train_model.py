import pickle

import pandas as pd

from psycourse.config import BLD_DATA, BLD_MODELS, BLD_RESULTS, SRC
from psycourse.ml_pipeline.train_model import svm_model

task_products = {
    "probabilities": BLD_DATA / "svm_predicted_probabilities.pkl",
    "model": BLD_MODELS / "svm_classifier.pkl",
    "learning_curve_data": BLD_RESULTS / "svm" / "learning_curve_data.pkl",
    "roc_data": BLD_RESULTS / "svm" / "roc_data.pkl",
    "nested_scores_data": BLD_RESULTS / "svm" / "nested_scores_data.pkl",
    "cm_data": BLD_RESULTS / "svm" / "cm_data.pkl",
    "report_dict": BLD_RESULTS / "svm" / "report_dict.pkl",
}


def task_train_model(
    script_path=SRC / "ml_pipeline" / "train_model.py",
    data_path=BLD_DATA / "sparse_dataset_with_targets.pkl",
    produces=task_products,
):
    """Train SVM Model and save the predicted probabilities and plot data."""

    data = pd.read_pickle(data_path)
    (
        final_model,
        cluster_probability_scores_df,
        learning_curve_data,
        roc_data,
        nested_scores_data,
        cm_data,
        report_dict,
    ) = svm_model(data)

    cluster_probability_scores_df.to_pickle(task_products["probabilities"])

    with open(task_products["model"], "wb") as f:
        pickle.dump(final_model, f)

    with open(task_products["learning_curve_data"], "wb") as f:
        pickle.dump(learning_curve_data, f)

    with open(task_products["roc_data"], "wb") as f:
        pickle.dump(roc_data, f)

    with open(task_products["nested_scores_data"], "wb") as f:
        pickle.dump(nested_scores_data, f)

    with open(task_products["cm_data"], "wb") as f:
        pickle.dump(cm_data, f)

    with open(task_products["report_dict"], "wb") as f:
        pickle.dump(report_dict, f)
