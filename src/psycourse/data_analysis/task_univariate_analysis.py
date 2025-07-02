import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_analysis import (
    univariate_lipid_class_regression,
    univariate_lipid_regression,
    univariate_lipid_regression_cov_diag,
    univariate_lipids_ancova,
    univariate_prs_ancova,
    univariate_prs_regression,
    univariate_prs_regression_cov_diag,
)

#### PRS DATA UNIVARIATE REGRESSION TASKS #####


def task_univariate_prs_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "univariate_prs_results.pkl",
):
    """Perform univariate regression on PRS data."""

    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results = univariate_prs_regression(data)
    univariate_prs_results.to_pickle(produces)


def task_univariate_prs_regression_cov_diag(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "univariate_prs_results_cov_diag.pkl",
):
    """Perform univariate regression on PRS data with diagnosis as added covariate."""

    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results = univariate_prs_regression_cov_diag(data)
    univariate_prs_results.to_pickle(produces)


def task_univariate_prs_ancova(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "univariate_prs_extremes_ancova_results.pkl",
):
    """Perform ANCOVA on PRS data focussing on extreme cases."""

    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results = univariate_prs_ancova(data)
    univariate_prs_results.to_pickle(produces)


##### LIPID DATA UNIVARIATE REGRESSION TASKS #####

task_univariate_lipid_regression_produces = {
    "top20_lipids": BLD_RESULTS / "univariate_lipid_results_top20.pkl",
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


def task_univariate_lipid_class_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "univariate_lipid_class_results.pkl",
):
    """Perform univariate regression on lipid data."""

    data = pd.read_pickle(multimodal_df_path)
    univariate_lipid_class_results = univariate_lipid_class_regression(data)
    univariate_lipid_class_results.to_pickle(produces)


task_univariate_lipid_ancova_produces = {
    "univariate_lipids_extremes_ancova_results": BLD_RESULTS
    / "univariate_lipids_extremes_ancova_results.pkl",
    "univariate_lipids_extremes_ancova_results_top20": BLD_RESULTS
    / "univariate_lipids_extremes_ancova_results_top20.pkl",
}


def task_univariate_lipid_ancova(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=task_univariate_lipid_ancova_produces,
):
    """Perform ANCOVA on PRS data focussing on extreme cases."""

    data = pd.read_pickle(multimodal_df_path)
    lipids_extremes_ancova_results, lipids_extremes_ancova_results_top20 = (
        univariate_lipids_ancova(data)
    )
    lipids_extremes_ancova_results.to_pickle(
        produces["univariate_lipids_extremes_ancova_results"]
    )
    lipids_extremes_ancova_results_top20.to_pickle(
        produces["univariate_lipids_extremes_ancova_results_top20"]
    )


### Covariate Diagnosis added to Univariate Lipid Regression ###

task_univariate_lipid_regression_produces_cov_diag = {
    "top20_lipids_cov_diag": BLD_RESULTS
    / "univariate_lipid_results_top20_cov_diag.pkl",
    "univariate_lipid_results_cov_diag": BLD_RESULTS
    / "univariate_lipid_results_cov_diag.pkl",
}


def task_univariate_lipid_regression_cov_diag(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=task_univariate_lipid_regression_produces_cov_diag,
):
    """Perform univariate regression on lipid data."""

    data = pd.read_pickle(multimodal_df_path)
    top20_lipids, univariate_lipid_results = univariate_lipid_regression_cov_diag(data)
    top20_lipids.to_pickle(produces["top20_lipids_cov_diag"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results_cov_diag"])
