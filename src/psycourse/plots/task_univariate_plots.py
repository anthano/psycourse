import matplotlib.pyplot as plt
import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.plots.univariate_plots import (
    plot_corr_matrix_lipid_classes,
    plot_corr_matrix_lipid_top20,
    plot_corr_matrix_prs,
    plot_perm_enrichment,
    plot_prs_cv_delta_mse,
    plot_univariate_lipid_class_regression,
    plot_univariate_lipid_extremes,
    plot_univariate_lipid_regression,
    plot_univariate_prs_extremes,
    plot_univariate_prs_regression,
)

############ Lipids ###################


def task_plot_univariate_lipid_regression(
    script_path=SRC / "plots" / "univariate_plots.py",
    top20_lipids_path=BLD_RESULTS / "univariate_lipid_results_top20.pkl",
    produces=BLD_RESULTS / "plots" / "univariate_lipid_regression_plot.svg",
):
    """Plot the top 20 lipids associated with cluster 5 probability
    using regression coefficients and FDR values."""

    lipid_top20 = pd.read_pickle(top20_lipids_path)
    fig, ax = plot_univariate_lipid_regression(lipid_top20)
    plt.savefig(produces, bbox_inches="tight")


def task_plot_univariate_lipid_class_regression(
    script_path=SRC / "plots" / "univariate_plots.py",
    lipid_class_results=BLD_RESULTS / "univariate_lipid_class_results.pkl",
    produces=BLD_RESULTS / "plots" / "univariate_lipid_class_regression_plot.svg",
):
    """Plot the top 20 lipids associated with cluster 5 probability
    using regression coefficients and FDR values."""

    lipid_class_results = pd.read_pickle(lipid_class_results)
    plot_univariate_lipid_class_regression(lipid_class_results)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
    plt.close()


def task_plot_univariate_lipid_extremes(
    script_path=SRC / "plots" / "univariate_plots.py",
    top20_lipids_path=BLD_RESULTS
    / "lipids"
    / "lipids_extremes_ancova_results_50_top20.pkl",
    produces=BLD_RESULTS / "plots" / "univariate_lipid_extremes_plot.svg",
):
    """Plot the top 20 lipids associated with cluster 5 probability
    using regression coefficients and FDR values."""

    lipid_top20 = pd.read_pickle(top20_lipids_path)
    plot_univariate_lipid_extremes(lipid_top20)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
    plt.close()


########################################################################################
# PRS
########################################################################################
def task_plot_univariate_prs_regression(
    script_path=SRC / "plots" / "univariate_plots.py",
    prs_results_path=BLD_RESULTS / "univariate_prs_results.pkl",
    produces=BLD_RESULTS / "plots" / "univariate_prs_regression_plot.svg",
):
    """Plot the prs associated with cluster 5 probability
    using regression coefficients and FDR values."""

    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")


def task_plot_univariate_prs_extremes(
    script_path=SRC / "plots" / "univariate_plots.py",
    prs_results_path=BLD_RESULTS / "univariate_prs_extremes_ancova_results.pkl",
    produces=BLD_RESULTS / "plots" / "univariate_prs_extremes.svg",
):
    """Plot the prs associated with cluster 5 probability
    using regression coefficients and FDR values for the top50 vs. bottom 50."""

    prs_results = pd.read_pickle(prs_results_path)
    plot_univariate_prs_extremes(prs_results)

    # Save the plot
    plt.savefig(produces)
    plt.close()


def task_plot_corr_matrix_lipid_top20(
    script_path=SRC / "plots" / "univariate_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    top20_lipids_path=BLD_RESULTS / "univariate_lipid_results_top20.pkl",
    produces=BLD_RESULTS / "plots" / "lipid_corr_matrix_top20.svg",
):
    """Plot the correlation matrix of the top 20 lipids."""

    multimodal_df = pd.read_pickle(multimodal_data_path)
    lipid_top20 = pd.read_pickle(top20_lipids_path)
    plot_corr_matrix_lipid_top20(multimodal_df, lipid_top20)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
    plt.close()


def task_plot_corr_matrix_lipid_class(
    script_path=SRC / "plots" / "univariate_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    lipid_class_results_path=BLD_RESULTS / "univariate_lipid_class_results.pkl",
    produces=BLD_RESULTS / "plots" / "lipid_class_corr_matrix.svg",
):
    """Plot the correlation matrix of the lipid classes."""

    multimodal_df = pd.read_pickle(multimodal_data_path)
    plot_corr_matrix_lipid_classes(multimodal_df)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
    plt.close()


def task_plot_corr_matrix_prs(
    script_path=SRC / "plots" / "univariate_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "plots" / "prs_corr_matrix.svg",
):
    """Plot the correlation matrix of the top 20 lipids."""

    multimodal_df = pd.read_pickle(multimodal_data_path)
    plot_corr_matrix_prs(multimodal_df)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
    plt.close()


def task_plot_prs_cv_delta_mse(
    script_path=SRC / "plots" / "univariate_plots.py",
    delta_df_path=BLD_RESULTS / "prs_cv_delta_mse_results.pkl",
    produces=BLD_RESULTS / "plots" / "prs_cv_delta_mse_plot.svg",
):
    """Plot the PRS cross-validated delta MSE results."""

    delta_df = pd.read_pickle(delta_df_path)
    fig, ax = plot_prs_cv_delta_mse(delta_df)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")


def task_plot_perm_enrichment(
    script_path=SRC / "plots" / "univariate_plots.py",
    enrich_df_path=BLD_RESULTS / "lipid_class_enrichment_perm_results.pkl",
    produces=BLD_RESULTS / "plots" / "lipid_class_enrichment_perm_plot.svg",
):
    """Plot the lipid class enrichment results with permutation testing."""

    enrich_df = pd.read_pickle(enrich_df_path)
    fig, ax = plot_perm_enrichment(enrich_df)

    # Save the plot
    plt.savefig(produces, bbox_inches="tight")
