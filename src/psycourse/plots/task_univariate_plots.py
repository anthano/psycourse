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
    / "univariate_lipid_regression_plot_standard_cov.png",
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
    / "univariate_lipid_regression_plot_cov_med.png",
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
    / "univariate_lipid_regression_plot_cov_diagnosis.png",
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
    / "univariate_lipid_regression_plot_cov_med_and_diag.png",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss.png",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_neg(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_neg.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_neg.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_neg.png",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_gen(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_gen.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_gen.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_gen.png",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_total_score(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_total_score.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_total_score.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_total_score.png",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_both(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_both.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_both.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_both.png",
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
    / "univariate_prs_regression_plot_standard_cov.png",
):
    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_prs_lipid_regression_combined_standard_cov(
    prs_results_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_standard_cov.pkl",
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_prs_lipid_regression_plot_combined_standard_cov.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_prs_lipid_regression_plot_combined_standard_cov.svg",
):
    prs_results = pd.read_pickle(prs_results_path)
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 13),
        constrained_layout=False,
    )

    # Reserve space on the left for the shared y-label
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.05, hspace=0.3)

    plot_univariate_prs_regression(prs_results, ax=axes[0])
    plot_univariate_lipid_regression(lipid_results, annotation_df, ax=axes[1])

    for ax, label in zip(axes, ["A", "B"], strict=False):
        ax.text(
            -0.12,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    for path in [bld_plots_dir_output, writing_plots_dir_output]:
        fig.savefig(path, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_prs_regression_cov_bmi(
    prs_results_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_cov_bmi.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_prs_regression_plot_cov_bmi.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "univariate_prs_regression_plot_cov_bmi.png",
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
    / "univariate_prs_regression_plot_cov_diagnosis.png",
):
    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()


# ============================================================================
# SENSITIVITY COMBINED TASKS
# ============================================================================


@task
def task_plot_sensitivity_combined_standard_cov(
    lip_std_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    lip_med_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med.pkl",
    lip_diag_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_diagnosis.pkl",
    lip_med_diag_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med_and_diag.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "sensitivity_combined_standard_cov.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "sensitivity_combined_standard_cov.svg",
):
    """Lipid regression sensitivity: standard covariate models.

    Layout: 2 rows × 2 columns (A4-friendly portrait).
      A (top-left)     Standard covariates
      B (top-right)    + Medication
      C (bottom-left)  + Diagnosis
      D (bottom-right) + Medication + Diagnosis
      Shared legend centred at the bottom of the figure.
    """
    lip_std = pd.read_pickle(lip_std_path)
    lip_med = pd.read_pickle(lip_med_path)
    lip_diag = pd.read_pickle(lip_diag_path)
    lip_med_diag = pd.read_pickle(lip_med_diag_path)
    annotation_df = pd.read_pickle(annotation_df_path)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 13))
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.95,
        bottom=0.09,
        hspace=0.15,
        wspace=0.48,
    )

    models = [
        (lip_std, "Standard covariates", axes[0, 0], "A"),
        (lip_med, "+ Medication", axes[0, 1], "B"),
        (lip_diag, "+ Diagnosis", axes[1, 0], "C"),
        (lip_med_diag, "+ Medication + Diagnosis", axes[1, 1], "D"),
    ]

    leg_handles, leg_labels = [], []
    for i, (df, title, ax, label) in enumerate(models):
        plot_univariate_lipid_regression(df, annotation_df, ax=ax)
        ax.set_title(title, fontsize=11, pad=6)
        ax.text(
            -0.15,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        # Capture handles from first panel, then remove all per-panel legends
        panel_leg = ax.get_legend()
        if panel_leg:
            if i == 0:
                leg_handles = panel_leg.legend_handles
                leg_labels = [t.get_text() for t in panel_leg.get_texts()]
            panel_leg.remove()

    # Single shared legend at the bottom of the figure
    if leg_handles:
        fig.legend(
            leg_handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(leg_handles),
            frameon=True,
            fontsize=11,
        )

    for path in [bld_plots_dir_output, writing_plots_dir_output]:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_sensitivity_combined_panss(
    lip_panss_pos_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss.pkl",
    lip_panss_neg_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_neg.pkl",
    lip_panss_gen_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_gen.pkl",
    lip_panss_tot_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_total_score.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "sensitivity_combined_panss.svg",
    writing_plots_dir_output: Annotated[Path, Product] = WRITING_PLOTS_DIR
    / "sensitivity_combined_panss.svg",
):
    """Lipid regression sensitivity: PANSS subscale covariate models.

    Layout: 2 rows × 2 columns (A4-friendly portrait).
      A (top-left)     + PANSS Positive
      B (top-right)    + PANSS Negative
      C (bottom-left)  + PANSS General
      D (bottom-right) + PANSS Total Score
      Shared legend centred at the bottom of the figure.
    """
    lip_panss_pos = pd.read_pickle(lip_panss_pos_path)
    lip_panss_neg = pd.read_pickle(lip_panss_neg_path)
    lip_panss_gen = pd.read_pickle(lip_panss_gen_path)
    lip_panss_tot = pd.read_pickle(lip_panss_tot_path)
    annotation_df = pd.read_pickle(annotation_df_path)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 13))
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.95,
        bottom=0.09,
        hspace=0.15,
        wspace=0.48,
    )

    models = [
        (lip_panss_pos, "+ PANSS Positive", axes[0, 0], "A"),
        (lip_panss_neg, "+ PANSS Negative", axes[0, 1], "B"),
        (lip_panss_gen, "+ PANSS General", axes[1, 0], "C"),
        (lip_panss_tot, "+ PANSS Total Score", axes[1, 1], "D"),
    ]

    leg_handles, leg_labels = [], []
    for i, (df, title, ax, label) in enumerate(models):
        plot_univariate_lipid_regression(df, annotation_df, ax=ax)
        ax.set_title(title, fontsize=11, pad=6)
        ax.text(
            -0.15,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        # Capture handles from first panel, then remove all per-panel legends
        panel_leg = ax.get_legend()
        if panel_leg:
            if i == 0:
                leg_handles = panel_leg.legend_handles
                leg_labels = [t.get_text() for t in panel_leg.get_texts()]
            panel_leg.remove()

    # Single shared legend at the bottom of the figure
    if leg_handles:
        fig.legend(
            leg_handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(leg_handles),
            frameon=True,
            fontsize=11,
        )

    for path in [bld_plots_dir_output, writing_plots_dir_output]:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
    / "lipid_corr_matrix_top20.png",
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
    / "prs_corr_matrix.png",
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
    / "prs_cv_delta_mse_plot.png",
):
    delta_df = pd.read_pickle(delta_df_path)
    fig, ax = plot_prs_cv_delta_mse(delta_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    fig.savefig(writing_plots_dir_output, bbox_inches="tight")
    plt.close()
