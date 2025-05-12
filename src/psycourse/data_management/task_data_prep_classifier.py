import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_classifier import (
    concatenate_features_and_targets,
    create_sparse_dataset_for_classifier,
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


task_sparse_dataset_with_targets_produces = {
    "old_with_targets": BLD_DATA / "sparse_dataset_with_targets.pkl",
    "new_with_targets": BLD_DATA / "new_sparse_dataset_without_targets.pkl",
}


def task_sparse_dataset_with_targets(
    script_path=SRC / "data_management" / "data_prep_classifier.py",
    phenotypic_data_path=BLD_DATA / "encoded_phenotypic_data.pkl",
    target_data_path=BLD_DATA / "cleaned_cluster_labels.pkl",
    produces=task_sparse_dataset_with_targets_produces,
):
    phenotypic_data = pd.read_pickle(phenotypic_data_path)
    target_data = pd.read_pickle(target_data_path)
    old_with_targets, new_with_targets = create_sparse_dataset_for_classifier(
        phenotypic_data, target_data
    )
    old_with_targets.to_pickle(produces["old_with_targets"])
    new_with_targets.to_pickle(produces["new_with_targets"])
