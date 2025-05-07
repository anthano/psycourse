"""Tasks for managing the data."""

import pandas as pd
import pytask

from psycourse.config import BLD_DATA, DATA_DIR, SRC
from psycourse.data_management.data_cleaning import (
    clean_cluster_labels,
    clean_lipidomic_data,
    clean_phenotypic_data,
)


@pytask.mark.wip
def task_clean_phenotypic_data(
    script_path=SRC / "data_management" / "data_cleaning.py",
    phenotypic_data_path=DATA_DIR / "230614_v6.0" / "230614_v6.0_psycourse_wd.csv",
    produces=BLD_DATA / "cleaned_phenotypic_data.pkl",
):
    """Clean the phenotypic data."""

    data = pd.read_csv(phenotypic_data_path, delimiter="\t", low_memory=False)
    cleaned_phenotypic_df = clean_phenotypic_data(data)
    cleaned_phenotypic_df.to_pickle(produces)


@pytask.mark.wip
def task_clean_lipidomic_data(
    script_path=SRC / "data_management" / "data_cleaning.py",
    lipid_intensities_path=DATA_DIR / "lipidomics" / "lipid_intensities.csv",
    sample_description_path=DATA_DIR / "lipidomics" / "sample_description.csv",
    produces=BLD_DATA / "cleaned_lipidomic_data.pkl",
):
    """Clean the lipidomic data."""

    lipid_intensities = pd.read_csv(lipid_intensities_path)
    sample_description = pd.read_csv(sample_description_path, delimiter=";")
    cleaned_lipidomic_df = clean_lipidomic_data(sample_description, lipid_intensities)
    cleaned_lipidomic_df.to_pickle(produces)


def task_clean_cluster_labels_data(
    script_path=SRC / "data_management" / "data_cleaning.py",
    cluster_labels_path=DATA_DIR / "ClusterLabels.csv",
    produces=BLD_DATA / "cleaned_cluster_labels.pkl",
):
    """Clean the cluster labels data."""

    cluster_labels = pd.read_csv(cluster_labels_path)
    cleaned_cluster_labels = clean_cluster_labels(cluster_labels)
    cleaned_cluster_labels.to_pickle(produces)
