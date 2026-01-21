import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_integrated_analysis import (
    prep_data_for_integrated_analysis,
)

OUTPUT_DIR = BLD_DATA / "integrated_analysis"

products = {
    "lipid_df_test": OUTPUT_DIR / "integrated_analysis_lipid_df_test.pkl",
    "lipid_class_df_test": OUTPUT_DIR / "integrated_analysis_lipid_class_df_test.pkl",
    "prs_df_test": OUTPUT_DIR / "integrated_analysis_prs_df_test.pkl",
    "outcome_df_test": OUTPUT_DIR / "integrated_analysis_outcome_df_test.pkl",
    "lipid_df_train": OUTPUT_DIR / "integrated_analysis_lipid_df_train.pkl",
    "lipid_class_df_train": OUTPUT_DIR / "integrated_analysis_lipid_class_df_train.pkl",
    "prs_df_train": OUTPUT_DIR / "integrated_analysis_prs_df_train.pkl",
    "outcome_df_train": OUTPUT_DIR / "integrated_analysis_outcome_df_train.pkl",
}


def task_prep_data_for_integrated_analysis(
    script_path=SRC / "data_management" / "data_prep_integrated_analysis.py",
    multimodal_lipid_subset_path=BLD_DATA / "multimodal_lipid_subset_df.pkl",
    produces=products,
):
    multimodal_lipid_subset_df = pd.read_pickle(multimodal_lipid_subset_path)
    (
        lipid_df_test,
        lipid_class_df_test,
        prs_df_test,
        outcome_df_test,
        lipid_df_train,
        lipid_class_df_train,
        prs_df_train,
        outcome_df_train,
    ) = prep_data_for_integrated_analysis(multimodal_lipid_subset_df)

    lipid_df_test.to_pickle(produces["lipid_df_test"])
    lipid_class_df_test.to_pickle(produces["lipid_class_df_test"])
    prs_df_test.to_pickle(produces["prs_df_test"])
    outcome_df_test.to_pickle(produces["outcome_df_test"])
    lipid_df_train.to_pickle(produces["lipid_df_train"])
    lipid_class_df_train.to_pickle(produces["lipid_class_df_train"])
    prs_df_train.to_pickle(produces["prs_df_train"])
    outcome_df_train.to_pickle(produces["outcome_df_train"])


products_csv = {
    "lipid_df_test": OUTPUT_DIR / "integrated_analysis_lipid_df_test.csv",
    "lipid_class_df_test": OUTPUT_DIR / "integrated_analysis_lipid_class_df_test.csv",
    "prs_df_test": OUTPUT_DIR / "integrated_analysis_prs_df_test.csv",
    "outcome_df_test": OUTPUT_DIR / "integrated_analysis_outcome_df_test.csv",
    "lipid_df_train": OUTPUT_DIR / "integrated_analysis_lipid_df_train.csv",
    "lipid_class_df_train": OUTPUT_DIR / "integrated_analysis_lipid_class_df_train.csv",
    "prs_df_train": OUTPUT_DIR / "integrated_analysis_prs_df_train.csv",
    "outcome_df_train": OUTPUT_DIR / "integrated_analysis_outcome_df_train.csv",
}


def task_prep_data_for_integrated_analysis_csv(
    script_path=SRC / "data_management" / "data_prep_integrated_analysis.py",
    multimodal_lipid_subset_path=BLD_DATA / "multimodal_lipid_subset_df.pkl",
    produces=products_csv,
):
    multimodal_lipid_subset_df = pd.read_pickle(multimodal_lipid_subset_path)
    (
        lipid_df_test,
        lipid_class_df_test,
        prs_df_test,
        outcome_df_test,
        lipid_df_train,
        lipid_class_df_train,
        prs_df_train,
        outcome_df_train,
    ) = prep_data_for_integrated_analysis(multimodal_lipid_subset_df)

    lipid_df_test.to_csv(produces["lipid_df_test"])
    lipid_class_df_test.to_csv(produces["lipid_class_df_test"])
    prs_df_test.to_csv(produces["prs_df_test"])
    outcome_df_test.to_csv(produces["outcome_df_test"])
    lipid_df_train.to_csv(produces["lipid_df_train"])
    lipid_class_df_train.to_csv(produces["lipid_class_df_train"])
    prs_df_train.to_csv(produces["prs_df_train"])
    outcome_df_train.to_csv(produces["outcome_df_train"])
