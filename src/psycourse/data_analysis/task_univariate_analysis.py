import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_analysis import (
    prs_cv_delta_mse,
    univariate_lipid_regression,
    univariate_lipid_regression_cov_diag,
    univariate_prs_ancova,
    univariate_prs_regression,
    univariate_prs_regression_cov_bmi,
    univariate_prs_regression_cov_diagnosis,
)

UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "prs"
)
UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid"
)

# ======================================================================================
# PRS TASKS
# ======================================================================================

univariate_prs_products = {
    "univariate_prs_results": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_standard_cov.pkl",
    "n_subset_dict": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_standard_cov.pkl",
}


def task_univariate_prs_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_prs_products,
):
    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results, n_subset_dict = univariate_prs_regression(data)
    univariate_prs_results.to_pickle(univariate_prs_products["univariate_prs_results"])
    pd.to_pickle(n_subset_dict, univariate_prs_products["n_subset_dict"])


# ======================================================================================
univariate_prs_products_cov_bmi = {
    "univariate_prs_results": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_cov_bmi.pkl",
    "n_subset_dict": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_bmi.pkl",
}


def task_univariate_prs_regression_cov_bmi(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR / "univariate_prs_results.pkl",
):
    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results, n_subset_dict = univariate_prs_regression_cov_bmi(data)
    univariate_prs_results.to_pickle(
        univariate_prs_products_cov_bmi["univariate_prs_results"]
    )
    pd.to_pickle(n_subset_dict, univariate_prs_products_cov_bmi["n_subset_dict"])


# =====================================================================================
univariate_prs_products_cov_diagnosis = {
    "univariate_prs_results": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_cov_diagnosis.pkl",
    "n_subset_dict": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_diagnosis.pkl",
}


def task_univariate_prs_regression_cov_diagnosis(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR / "univariate_prs_results.pkl",
):
    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results, n_subset_dict = univariate_prs_regression_cov_diagnosis(
        data
    )
    univariate_prs_results.to_pickle(
        univariate_prs_products_cov_diagnosis["univariate_prs_results"]
    )
    pd.to_pickle(n_subset_dict, univariate_prs_products_cov_diagnosis["n_subset_dict"])


task_univariate_prs_ancova_produces = {
    "prs_extremes_ancova_results[50]": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_extremes_ancova_results_50.pkl",
    "prs_extremes_ancova_results[100]": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_extremes_ancova_results_100.pkl",
    "prs_extremes_ancova_results[120]": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
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
    produces=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR / "prs_cv_delta_mse_results.pkl",
):
    """Perform cross-validated delta MSE analysis on PRS data."""

    data = pd.read_pickle(multimodal_df_path)
    prs_cv_delta_mse_results = prs_cv_delta_mse(data)
    prs_cv_delta_mse_results.to_pickle(produces)


# ======================================================================================
# LIPID TASKS
# ======================================================================================

task_univariate_lipid_regression_produces = {
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results.pkl",
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


### Covariate Diagnosis added to Univariate Lipid Regression ###

task_univariate_lipid_regression_produces_cov_diag = {
    "top20_lipids_cov_diag": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_diag.pkl",
    "univariate_lipid_results_cov_diag": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
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
