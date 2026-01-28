from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product, task

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC, WRITING
from psycourse.plots.univariate_plots import (
    plot_corr_matrix_lipid_top20,
    plot_corr_matrix_prs,
    plot_prs_cv_delta_mse,
    plot_univariate_lipid_regression,
    plot_univariate_prs_regression,
)

UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "prs"
)
UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid"
)
BLD_PLOTS_DIR = BLD_RESULTS / "plots" / "univariate_analysis"
WRITING_PLOTS_DIR = WRITING / "plots" / "univariate_analysis"

ANNOTATION_DF_PATH = BLD_DATA / "cleaned_lipid_class_data.pkl"

# ======================================================================================
# LIPIDS
# ======================================================================================
FILES = {
    "standard_cov": "univariate_lipid_results_top20.pkl",
    "cov_med": "univariate_lipid_results_top20_cov_med.pkl",
    "cov_diagnosis": "univariate_lipid_results_top20_cov_diagnosis.pkl",
    "cov_med_and_diag": "univariate_lipid_results_top20_cov_med_and_diag.pkl",
}

for name, input_file in FILES.items():
    input_path = UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR / input_file
    bld_output_path = BLD_PLOTS_DIR / f"univariate_lipid_regression_plot_{name}.svg"
    writing_output_path = WRITING_PLOTS_DIR / bld_output_path.name

    @task(
        id=name,
        kwargs={
            "lipid_results_path": input_path,
            "annotation_df_path": ANNOTATION_DF_PATH,
            "produces": [bld_output_path, writing_output_path],
        },
    )
    def task_plot_univariate_lipid_regression(
        lipid_results_path: Path,
        annotation_df_path: Path,
        produces: Annotated[list[Path], Product],
    ):
        lipid_results = pd.read_pickle(lipid_results_path)
        annotation_df = pd.read_pickle(annotation_df_path)
        fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
        for path in produces:
            plt.savefig(path, bbox_inches="tight")


# ======================================================================================
# PRS
# ======================================================================================
FILES = {
    "standard_cov": "univariate_prs_results_standard_cov.pkl",
    "cov_bmi": "univariate_prs_results_cov_bmi.pkl",
    "cov_diagnosis": "univariate_prs_results_cov_diagnosis.pkl",
}

for name, input_file in FILES.items():
    input_path = UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR / input_file
    bld_output_path = BLD_PLOTS_DIR / f"univariate_prs_regression_plot_{name}.svg"
    writing_output_path = WRITING_PLOTS_DIR / bld_output_path.name

    @task(
        id=name,
        kwargs={
            "prs_results_path": input_path,
            "produces": [bld_output_path, writing_output_path],
        },
    )
    def task_plot_univariate_prs_regression(
        prs_results_path: Path,
        produces: Annotated[list[Path], Product],
    ):
        prs_results = pd.read_pickle(prs_results_path)
        fig, ax = plot_univariate_prs_regression(prs_results)
        for path in produces:
            plt.savefig(path, bbox_inches="tight")


# ======================================================================================
# Correlation Matrix - Lipids
# ======================================================================================
@task
def task_plot_corr_matrix_lipid_top20(
    script_path=SRC / "plots" / "univariate_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    top20_lipids_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    annotation_df=BLD_DATA / "cleaned_lipid_class_data.pkl",
    produces: Annotated[list[Path], Product] = None,
):
    if produces is None:
        produces = [
            BLD_PLOTS_DIR / "lipid_corr_matrix_top20.svg",
            WRITING_PLOTS_DIR / "lipid_corr_matrix_top20.svg",
        ]

    multimodal_df = pd.read_pickle(multimodal_data_path)
    lipid_top20 = pd.read_pickle(top20_lipids_path)
    annotation_df = pd.read_pickle(annotation_df)

    plot_corr_matrix_lipid_top20(multimodal_df, lipid_top20, annotation_df)

    for path in produces:
        plt.savefig(path, bbox_inches="tight")
    plt.close()


# ======================================================================================
# Correlation Matrix - PRS
# ======================================================================================
@task
def task_plot_corr_matrix_prs(
    script_path=SRC / "plots" / "univariate_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces: Annotated[list[Path], Product] = None,
):
    if produces is None:
        produces = [
            BLD_PLOTS_DIR / "prs_corr_matrix.svg",
            WRITING_PLOTS_DIR / "prs_corr_matrix.svg",
        ]

    multimodal_df = pd.read_pickle(multimodal_data_path)
    plot_corr_matrix_prs(multimodal_df)

    for path in produces:
        plt.savefig(path, bbox_inches="tight")
    plt.close()


# ======================================================================================
# PRS Delta MSE
# ======================================================================================
@task
def task_plot_prs_cv_delta_mse(
    script_path=SRC / "plots" / "univariate_plots.py",
    delta_df_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_cv_delta_mse_results.pkl",
    produces: Annotated[list[Path], Product] = None,
):
    if produces is None:
        produces = [
            BLD_PLOTS_DIR / "prs_cv_delta_mse_plot.svg",
            WRITING_PLOTS_DIR / "prs_cv_delta_mse_plot.svg",
        ]

    delta_df = pd.read_pickle(delta_df_path)
    fig, ax = plot_prs_cv_delta_mse(delta_df)

    for path in produces:
        plt.savefig(path, bbox_inches="tight")
