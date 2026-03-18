import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_analysis import merge_multimodal_complete_df

# PRS columns present in multimodal_complete_df (from clean_prs_data).
# A participant has PRS data if any of these is non-null.
PRS_COLS = [
    "ADHD_PRS",
    "ASD_PRS",
    "Education_PRS",
    "SCZ_PRS",
    "Agreeableness_PRS",
    "Alzheimer_PRS",
    "Conscientiousness_PRS",
    "Extraversion_PRS",
    "MDD_PRS",
    "Neuroticism_PRS",
    "Openness_PRS",
    "SleepDurationLong_PRS",
    "SleepDurationShort_PRS",
    "BD_PRS",
]


def task_create_prs_subset_df(
    script_path=SRC / "data_management" / "task_data_prep_analysis.py",
    multimodal_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_DATA / "multimodal_prs_subset_df.pkl",
):
    """Filter the full multimodal dataframe to participants with PRS data."""
    df = pd.read_pickle(multimodal_path)
    prs_cols_present = [c for c in PRS_COLS if c in df.columns]
    prs_df = df.dropna(subset=prs_cols_present, how="all")
    prs_df.to_pickle(produces)


def task_merge_multimodal_complete_df(
    script_path=SRC / "data_management" / "data_prep_analysis.py",
    lipid_data_path=BLD_DATA / "cleaned_lipidomic_data.pkl",
    lipid_class_path=BLD_DATA / "cleaned_lipid_class_data.pkl",
    phenotypic_data_path=BLD_DATA / "cleaned_phenotypic_data.pkl",
    cluster_probabilities_full_path=BLD_DATA / "svm_predicted_probabilities_full.pkl",
    prs_data_path=BLD_DATA / "cleaned_prs_data.pkl",
    pc_data_path=BLD_DATA / "cleaned_pc_data.pkl",
    produces=BLD_DATA / "multimodal_complete_df.pkl",
):
    """Merge lipid data, phenotypic data, cluster probabilities, and PRS data into a
    single dataframe."""

    lipid_data = pd.read_pickle(lipid_data_path)
    lipid_class_data = pd.read_pickle(lipid_class_path)
    phenotypic_data = pd.read_pickle(phenotypic_data_path)
    cluster_probabilities_full = pd.read_pickle(cluster_probabilities_full_path)
    prs_data = pd.read_pickle(prs_data_path)
    pc_data = pd.read_pickle(pc_data_path)

    multimodal_df = merge_multimodal_complete_df(
        lipid_data,
        lipid_class_data,
        phenotypic_data,
        cluster_probabilities_full,
        prs_data,
        pc_data,
    )

    multimodal_df.to_pickle(produces)
