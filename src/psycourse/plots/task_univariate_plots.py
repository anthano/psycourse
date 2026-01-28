from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product, task

from psycourse.config import BLD_DATA, BLD_RESULTS, WRITING
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


# ============================================================================
# UNIVARIATE LIPID REGRESSION TASKS
# ============================================================================


@task
def task_plot_univariate_lipid_regression_standard_cov(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_standard_cov.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_standard_cov.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_med(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_med.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_med.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_diagnosis(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_diagnosis.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_diagnosis.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_diagnosis.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_med_and_diag(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med_and_diag.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_med_and_diag.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_med_and_diag.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


# ============================================================================
# UNIVARIATE PRS REGRESSION TASKS
# ============================================================================


@task
def task_plot_univariate_prs_regression_standard_cov(
    prs_results_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_standard_cov.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_prs_regression_plot_standard_cov.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_prs_regression_plot_standard_cov.svg",
):
    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_prs_regression_cov_bmi(
    prs_results_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_cov_bmi.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_prs_regression_plot_cov_bmi.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_prs_regression_plot_cov_bmi.svg",
):
    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_prs_regression_cov_diagnosis(
    prs_results_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_cov_diagnosis.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_prs_regression_plot_cov_diagnosis.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_prs_regression_plot_cov_diagnosis.svg",
):
    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


# ============================================================================
# CORR MATRIX & DELTA MSE TASKS
# ============================================================================


@task
def task_plot_corr_matrix_lipid_top20(
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    top20_lipids_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "lipid_corr_matrix_top20.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "lipid_corr_matrix_top20.svg",
):
    multimodal_df = pd.read_pickle(multimodal_data_path)
    lipid_top20 = pd.read_pickle(top20_lipids_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    plot_corr_matrix_lipid_top20(multimodal_df, lipid_top20, annotation_df)
    plt.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_corr_matrix_prs(
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "prs_corr_matrix.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "prs_corr_matrix.svg",
):
    multimodal_df = pd.read_pickle(multimodal_data_path)
    plot_corr_matrix_prs(multimodal_df)
    plt.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_prs_cv_delta_mse(
    delta_df_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_cv_delta_mse_results.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "prs_cv_delta_mse_plot.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "prs_cv_delta_mse_plot.svg",
):
    delta_df = pd.read_pickle(delta_df_path)
    fig, ax = plot_prs_cv_delta_mse(delta_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()
