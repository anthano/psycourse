import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_classifier import (
    concatenate_features_and_targets,
)


def task_concatenate_features_and_targets(
    script_path=SRC / "data_management" / "data_prep_classifier.py",
    phenotypic_data_path=BLD_DATA / "encoded_phenotypic_data.pkl",
    target_data_path=BLD_DATA / "cleaned_cluster_labels.pkl",
    produces=BLD_DATA / "full_dataset_for_classifier.pkl",
):
    phenotypic_data = pd.read_pickle(phenotypic_data_path)
    target_data = pd.read_pickle(target_data_path)
    full_dataset = concatenate_features_and_targets(phenotypic_data, target_data)
    full_dataset.to_pickle(produces)
