import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_analysis import merge_multimodal_complete_df


def task_merge_multimodal_complete_df(
    script_path=SRC / "data_management" / "data_prep_analysis.py",
    lipid_data_path=BLD_DATA / "cleaned_lipidomic_data.pkl",
    phenotypic_data_path=BLD_DATA / "cleaned_phenotypic_data.pkl",
    cluster_probabilities_full_path=BLD_DATA / "svm_predicted_probabilities_full.pkl",
    prs_data_path=BLD_DATA / "cleaned_prs_data.pkl",
    produces=BLD_DATA / "multimodal_complete_df.pkl",
):
    """Merge lipid data, phenotypic data, cluster probabilities, and PRS data into a
    single dataframe."""

    lipid_data = pd.read_pickle(lipid_data_path)
    phenotypic_data = pd.read_pickle(phenotypic_data_path)
    cluster_probabilities_full = pd.read_pickle(cluster_probabilities_full_path)
    prs_data = pd.read_pickle(prs_data_path)

    multimodal_df = merge_multimodal_complete_df(
        lipid_data, phenotypic_data, cluster_probabilities_full, prs_data
    )

    multimodal_df.to_pickle(produces)
