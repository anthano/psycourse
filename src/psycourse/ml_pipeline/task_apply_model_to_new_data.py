import pickle

import pandas as pd

from psycourse.config import BLD_DATA, BLD_MODELS, SRC
from psycourse.ml_pipeline.apply_model_to_new_data import apply_model_to_new_data


def task_apply_model_to_new_data(
    script_path=SRC / "ml_pipeline" / "apply_model_to_new_data.py",
    data_path=BLD_DATA / "new_sparse_dataset_without_targets.pkl",
    model_path=BLD_MODELS / "svm_classifier.pkl",
    produces=BLD_DATA / "svm_predicted_probabilities_new_psycourse_data.pkl",
):
    data = pd.read_pickle(data_path)
    svm_model = pickle.load(open(model_path, "rb"))
    cluster_probability_scores_df = apply_model_to_new_data(svm_model, data)
    cluster_probability_scores_df.to_pickle(produces)
