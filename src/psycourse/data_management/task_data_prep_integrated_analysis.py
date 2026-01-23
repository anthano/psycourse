import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_integrated_analysis import (
    prep_data_for_integrated_analysis,
)

OUTPUT_DIR = BLD_DATA / "integrated_analysis"

# Define base filenames
base_names = [
    "lipid_df_train",
    "lipid_df_test",
    "lipid_class_df_train",
    "lipid_class_df_test",
    "prs_df_train",
    "prs_df_test",
    "outcome_df_train",
    "outcome_df_test",
    "prs_cov_train",
    "prs_cov_test",
    "lipid_cov_train",
    "lipid_cov_test",
]

products_pkl = {k: OUTPUT_DIR / f"integrated_analysis_{k}.pkl" for k in base_names}
products_csv = {k: OUTPUT_DIR / f"integrated_analysis_{k}.csv" for k in base_names}


def task_prep_data_for_integrated_analysis(
    script_path=SRC / "data_management" / "data_prep_integrated_analysis.py",
    multimodal_lipid_subset_path=BLD_DATA / "multimodal_lipid_subset_df.pkl",
    produces=products_pkl,
):
    multimodal_lipid_subset_df = pd.read_pickle(multimodal_lipid_subset_path)
    prepared = prep_data_for_integrated_analysis(multimodal_lipid_subset_df)

    for key, path in produces.items():
        df = _extract_data(prepared, key)
        df.to_pickle(path)


def task_prep_data_for_integrated_analysis_csv(
    script_path=SRC / "data_management" / "data_prep_integrated_analysis.py",
    multimodal_lipid_subset_path=BLD_DATA / "multimodal_lipid_subset_df.pkl",
    produces=products_csv,
):
    multimodal_lipid_subset_df = pd.read_pickle(multimodal_lipid_subset_path)
    prepared = prep_data_for_integrated_analysis(multimodal_lipid_subset_df)

    for key, path in produces.items():
        df = _extract_data(prepared, key)
        df.to_csv(path, index=False)


def _extract_data(prepared, key):
    mapping = {
        "lipid_df_train": "lipid_train",
        "lipid_df_test": "lipid_test",
        "lipid_class_df_train": "lipid_class_train",
        "lipid_class_df_test": "lipid_class_test",
        "prs_df_train": "prs_train",
        "prs_df_test": "prs_test",
        "outcome_df_train": "y_train",
        "outcome_df_test": "y_test",
        "prs_cov_train": "prs_cov_train",
        "prs_cov_test": "prs_cov_test",
        "lipid_cov_train": "lipid_cov_train",
        "lipid_cov_test": "lipid_cov_test",
    }
    return prepared[mapping[key]]
