import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_analysis import (
    lipid_class_enrichment_perm,
    prs_cv_delta_mse,
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


task_univariate_prs_ancova_produces = {
    "prs_extremes_ancova_results[50]": BLD_RESULTS
    / "prs"
    / "prs_extremes_ancova_results_50.pkl",
    "prs_extremes_ancova_results[100]": BLD_RESULTS
    / "prs"
    / "prs_extremes_ancova_results_100.pkl",
    "prs_extremes_ancova_results[120]": BLD_RESULTS
    / "prs"
    / "prs_extremes_ancova_results_120.pkl",
}


def task_univariate_prs_ancova(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=task_univariate_prs_ancova_produces,
):
    """Perform ANCOVA on PRS data focussing on extreme cases."""

    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_ancova_results = univariate_prs_ancova(data)
    univariate_prs_ancova_results[50].to_pickle(
        produces["prs_extremes_ancova_results[50]"]
    )
    univariate_prs_ancova_results[100].to_pickle(
        produces["prs_extremes_ancova_results[100]"]
    )
    univariate_prs_ancova_results[120].to_pickle(
        produces["prs_extremes_ancova_results[120]"]
    )


def task_prs_cv_delta_mse(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "prs_cv_delta_mse_results.pkl",
):
    """Perform cross-validated delta MSE analysis on PRS data."""

    data = pd.read_pickle(multimodal_df_path)
    prs_cv_delta_mse_results = prs_cv_delta_mse(data)
    prs_cv_delta_mse_results.to_pickle(produces)


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
    "lipids_ancova_results[50]": BLD_RESULTS
    / "lipids"
    / "lipids_extremes_ancova_results_50.pkl",
    "lipids_ancova_results[100]": BLD_RESULTS
    / "lipids"
    / "lipids_extremes_ancova_results_100.pkl",
    "lipids_ancova_results[120]": BLD_RESULTS
    / "lipids"
    / "lipids_extremes_ancova_results_120.pkl",
    "lipids_ancova_results_top20[50]": BLD_RESULTS
    / "lipids"
    / "lipids_extremes_ancova_results_50_top20.pkl",
    "lipids_ancova_results_top20[100]": BLD_RESULTS
    / "lipids"
    / "lipids_extremes_ancova_results_100_top20.pkl",
    "lipids_ancova_results_top20[120]": BLD_RESULTS
    / "lipids"
    / "lipids_extremes_ancova_results_120_top20.pkl",
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
    lipids_extremes_ancova_results[50].to_pickle(produces["lipids_ancova_results[50]"])
    lipids_extremes_ancova_results[100].to_pickle(
        produces["lipids_ancova_results[100]"]
    )
    lipids_extremes_ancova_results[120].to_pickle(
        produces["lipids_ancova_results[120]"]
    )
    lipids_extremes_ancova_results_top20[50].to_pickle(
        produces["lipids_ancova_results_top20[50]"]
    )
    lipids_extremes_ancova_results_top20[100].to_pickle(
        produces["lipids_ancova_results_top20[100]"]
    )
    lipids_extremes_ancova_results_top20[120].to_pickle(
        produces["lipids_ancova_results_top20[120]"]
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


def task_lipid_class_enrichment_perm(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    lipid_results=BLD_RESULTS / "univariate_lipid_results.pkl",
    lipid_class_df=BLD_DATA / "cleaned_lipid_class_data.pkl",
    produces=BLD_RESULTS / "lipid_class_enrichment_perm_results.pkl",
):
    """Perform lipid class enrichment analysis with permutation testing."""

    lipid_results = pd.read_pickle(lipid_results)
    lipid_class_df = pd.read_pickle(lipid_class_df)
    lipid_class_enrichment_results = lipid_class_enrichment_perm(
        lipid_results, lipid_class_df
    )
    lipid_class_enrichment_results.to_pickle(produces)
