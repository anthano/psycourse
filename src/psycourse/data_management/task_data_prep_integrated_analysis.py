import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_integrated_analysis import (
    prep_data_for_integrated_analysis,
)

products = {
    "lipid_data": BLD_DATA / "integrated_analysis_lipid_data.pkl",
    "prs_data": BLD_DATA / "integrated_analysis_prs_data.pkl",
    "outcome_data": BLD_DATA / "integrated_analysis_outcome_data.pkl",
}


def task_prep_data_for_integrated_analysis(
    script_path=SRC / "data_management" / "data_prep_integrated_analysis.py",
    multimodal_lipid_subset_path=BLD_DATA / "multimodal_lipid_subset_df.pkl",
    produces=products,
):
    multimodal_lipid_subset_df = pd.read_pickle(multimodal_lipid_subset_path)
    lipid_df, prs_df, outcome_df = prep_data_for_integrated_analysis(
        multimodal_lipid_subset_df
    )
    lipid_df.to_pickle(produces["lipid_data"])
    prs_df.to_pickle(produces["prs_data"])
    outcome_df.to_pickle(produces["outcome_data"])
