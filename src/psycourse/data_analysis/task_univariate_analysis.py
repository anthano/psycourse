import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_analysis import (
    univariate_lipid_regression,
    univariate_prs_regression,
)


def task_univariate_prs_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "univariate_prs_results.pkl",
):
    """Perform univariate regression on PRS data."""

    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results = univariate_prs_regression(data)
    univariate_prs_results.to_pickle(produces)


task_univariate_lipid_regression_produces = {
    "top20_lipids": BLD_RESULTS / "univariate_lipid_results top20.pkl",
    "univariate_lipid_results": BLD_RESULTS / "univariate_lipid_results.pkl",
}


def task_univariate_lipid_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=task_univariate_lipid_regression_produces,
):
    """Perform univariate regression on lipid data."""

    data = pd.read_pickle(multimodal_df_path)
    top20_lipids, univariate_lipid_results = univariate_lipid_regression(data)
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])
