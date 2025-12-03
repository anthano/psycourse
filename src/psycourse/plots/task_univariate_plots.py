import matplotlib.pyplot as plt
import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
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
BLD_PLOTS_DIR = BLD_RESULTS / "plots"


# ======================================================================================
# LIPIDS
# ======================================================================================
def task_plot_univariate_lipid_regression(
    script_path=SRC / "plots" / "univariate_plots.py",
    top20_lipids_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    annotation_df_path=BLD_DATA / "cleaned_lipid_class_data.pkl",
    produces=BLD_PLOTS_DIR / "univariate_lipid_regression_plot.svg",
):
    """Plot the top 20 lipids associated with cluster 5 probability
    using regression coefficients and FDR values."""

    lipid_top20 = pd.read_pickle(top20_lipids_path)
    cleaned_annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_top20, cleaned_annotation_df)
    plt.savefig(produces, bbox_inches="tight")


########################################################################################
# PRS
########################################################################################
def task_plot_univariate_prs_regression(
    script_path=SRC / "plots" / "univariate_plots.py",
    prs_results_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results.pkl",
    produces=BLD_PLOTS_DIR / "univariate_prs_regression_plot.svg",
):
    """Plot the prs associated with cluster 5 probability
    using regression coefficients and FDR values."""

    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")


def task_plot_corr_matrix_lipid_top20(
    script_path=SRC / "plots" / "univariate_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    top20_lipids_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    annotation_df=BLD_DATA / "cleaned_lipid_class_data.pkl",
    produces=BLD_PLOTS_DIR / "lipid_corr_matrix_top20.svg",
):
    """Plot the correlation matrix of the top 20 lipids."""

    multimodal_df = pd.read_pickle(multimodal_data_path)
    lipid_top20 = pd.read_pickle(top20_lipids_path)
    annotation_df = pd.read_pickle(annotation_df)
    plot_corr_matrix_lipid_top20(multimodal_df, lipid_top20, annotation_df)
    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
    plt.close()


def task_plot_corr_matrix_prs(
    script_path=SRC / "plots" / "univariate_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_PLOTS_DIR / "prs_corr_matrix.svg",
):
    """Plot the correlation matrix of the top 20 lipids."""

    multimodal_df = pd.read_pickle(multimodal_data_path)
    plot_corr_matrix_prs(multimodal_df)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
    plt.close()


def task_plot_prs_cv_delta_mse(
    script_path=SRC / "plots" / "univariate_plots.py",
    delta_df_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_cv_delta_mse_results.pkl",
    produces=BLD_PLOTS_DIR / "prs_cv_delta_mse_plot.svg",
):
    """Plot the PRS cross-validated delta MSE results."""

    delta_df = pd.read_pickle(delta_df_path)
    fig, ax = plot_prs_cv_delta_mse(delta_df)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
