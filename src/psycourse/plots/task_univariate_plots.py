from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product, task

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.plots.lipid_enrichment_plot import plot_lipid_coef_distributions
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
UNIVARIATE_LIPID_PANSS_DIR = UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR / "panss"
UNIVARIATE_LIPID_MED_ADJ_DIR = (
    UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR / "medication_adjusted"
)
BLD_PLOTS_DIR = BLD_RESULTS / "plots" / "univariate_analysis"
BLD_ENRICHMENT_PLOT_DIR = BLD_RESULTS / "plots" / "enrichment_analysis"

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
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_med(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_med.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_diagnosis(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_diagnosis.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_diagnosis.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_med_and_diag(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med_and_diag.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_med_and_diag.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_neg(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_neg.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_neg.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_gen(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_gen.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_gen.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_total_score(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_total_score.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_total_score.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_panss_both(
    lipid_results_path=UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_both.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_panss_both.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
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
):
    prs_results = pd.read_pickle(prs_results_path)
    fig, ax = plot_univariate_prs_regression(prs_results)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
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

    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
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

    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
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

    plt.close(fig)


# ============================================================================
# MEDICATION-ADJUSTED LIPID REGRESSION TASKS
# ============================================================================


@task
def task_plot_univariate_lipid_regression_cov_antidepressants(
    lipid_results_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_antidepressants.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_antidepressants.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_antipsychotics(
    lipid_results_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_antipsychotics.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_antipsychotics.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_tranquilizers(
    lipid_results_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_tranquilizers.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_tranquilizers.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_univariate_lipid_regression_cov_mood_stabilizers(
    lipid_results_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_mood_stabilizers.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "univariate_lipid_regression_plot_cov_mood_stabilizers.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_sensitivity_combined_medication_adjusted(
    lip_antidep_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_antidepressants.pkl",
    lip_antipsych_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_antipsychotics.pkl",
    lip_tranq_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_tranquilizers.pkl",
    lip_mood_path=UNIVARIATE_LIPID_MED_ADJ_DIR
    / "univariate_lipid_results_top20_cov_mood_stabilizers.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "sensitivity_combined_medication_adjusted.svg",
):
    """2×2 forest plot: each medication class controlled individually."""
    lip_antidep = pd.read_pickle(lip_antidep_path)
    lip_antipsych = pd.read_pickle(lip_antipsych_path)
    lip_tranq = pd.read_pickle(lip_tranq_path)
    lip_mood = pd.read_pickle(lip_mood_path)
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
        (lip_antidep, "+ Antidepressants", axes[0, 0], "A"),
        (lip_antipsych, "+ Antipsychotics", axes[0, 1], "B"),
        (lip_tranq, "+ Tranquilizers", axes[1, 0], "C"),
        (lip_mood, "+ Mood stabilizers", axes[1, 1], "D"),
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
        panel_leg = ax.get_legend()
        if panel_leg:
            if i == 0:
                leg_handles = panel_leg.legend_handles
                leg_labels = [t.get_text() for t in panel_leg.get_texts()]
            panel_leg.remove()

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
):
    multimodal_df = pd.read_pickle(multimodal_data_path)
    lipid_top20 = pd.read_pickle(top20_lipids_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    plot_corr_matrix_lipid_top20(multimodal_df, lipid_top20, annotation_df)
    plt.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_corr_matrix_prs(
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "prs_corr_matrix.svg",
):
    multimodal_df = pd.read_pickle(multimodal_data_path)
    plot_corr_matrix_prs(multimodal_df)
    plt.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


@task
def task_plot_prs_cv_delta_mse(
    delta_df_path=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_cv_delta_mse_results.pkl",
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "prs_cv_delta_mse_plot.svg",
):
    delta_df = pd.read_pickle(delta_df_path)
    fig, ax = plot_prs_cv_delta_mse(delta_df)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close()


# ============================================================================
# PANSS OUTCOME TASKS  (Lipid regression & enrichment with PANSS as outcome)
# ============================================================================

_PANSS_LABELS = {
    "pos": "PANSS Positive",
    "neg": "PANSS Negative",
    "gen": "PANSS General",
    "tot": "PANSS Total Score",
}

# ── Individual lipid regression plots ────────────────────────────────────────


@task
def task_plot_lipid_regression_panss_outcome_pos(
    lipid_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_sum_pos.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "lipid_regression_panss_outcome_pos.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    ax.set_title(_PANSS_LABELS["pos"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_lipid_regression_panss_outcome_neg(
    lipid_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_sum_neg.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "lipid_regression_panss_outcome_neg.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    ax.set_title(_PANSS_LABELS["neg"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_lipid_regression_panss_outcome_gen(
    lipid_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_sum_gen.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "lipid_regression_panss_outcome_gen.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    ax.set_title(_PANSS_LABELS["gen"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_lipid_regression_panss_outcome_tot(
    lipid_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_total_score.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "lipid_regression_panss_outcome_tot.svg",
):
    lipid_results = pd.read_pickle(lipid_results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    fig, ax = plot_univariate_lipid_regression(lipid_results, annotation_df)
    ax.set_title(_PANSS_LABELS["tot"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


# ── Individual enrichment plots ───────────────────────────────────────────────


@task
def task_plot_lipid_enrichment_panss_outcome_pos(
    results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_sum_pos.pkl",
    enrich_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_sum_pos.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_ENRICHMENT_PLOT_DIR
    / "lipid_enrichment_panss_outcome_pos.svg",
):
    results_df = pd.read_pickle(results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    enrich_df = pd.read_pickle(enrich_results_path)
    fig, ax = plot_lipid_coef_distributions(results_df, annotation_df, enrich_df)
    ax.set_title(_PANSS_LABELS["pos"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_lipid_enrichment_panss_outcome_neg(
    results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_sum_neg.pkl",
    enrich_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_sum_neg.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_ENRICHMENT_PLOT_DIR
    / "lipid_enrichment_panss_outcome_neg.svg",
):
    results_df = pd.read_pickle(results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    enrich_df = pd.read_pickle(enrich_results_path)
    fig, ax = plot_lipid_coef_distributions(results_df, annotation_df, enrich_df)
    ax.set_title(_PANSS_LABELS["neg"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_lipid_enrichment_panss_outcome_gen(
    results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_sum_gen.pkl",
    enrich_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_gen.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_ENRICHMENT_PLOT_DIR
    / "lipid_enrichment_panss_outcome_gen.svg",
):
    results_df = pd.read_pickle(results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    enrich_df = pd.read_pickle(enrich_results_path)
    fig, ax = plot_lipid_coef_distributions(results_df, annotation_df, enrich_df)
    ax.set_title(_PANSS_LABELS["gen"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_lipid_enrichment_panss_outcome_tot(
    results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_total_score.pkl",
    enrich_results_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_total_score.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_ENRICHMENT_PLOT_DIR
    / "lipid_enrichment_panss_outcome_tot.svg",
):
    results_df = pd.read_pickle(results_path)
    annotation_df = pd.read_pickle(annotation_df_path)
    enrich_df = pd.read_pickle(enrich_results_path)
    fig, ax = plot_lipid_coef_distributions(results_df, annotation_df, enrich_df)
    ax.set_title(_PANSS_LABELS["tot"], fontsize=12)
    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


# ── Combined 2×2 figures ──────────────────────────────────────────────────────


@task
def task_plot_panss_outcome_lipid_regression_combined(
    lip_pos_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_sum_pos.pkl",
    lip_neg_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_sum_neg.pkl",
    lip_gen_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_sum_gen.pkl",
    lip_tot_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_top20_standard_panss_total_score.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_PLOTS_DIR
    / "panss_outcome_lipid_regression_combined.svg",
):
    """Combined lipid regression figure with each PANSS subscale as outcome.

    Layout: 2 rows × 2 columns (A4-friendly portrait).
      A (top-left)     PANSS Positive
      B (top-right)    PANSS Negative
      C (bottom-left)  PANSS General
      D (bottom-right) PANSS Total Score
      Shared legend centred at the bottom.
    """
    lip_pos = pd.read_pickle(lip_pos_path)
    lip_neg = pd.read_pickle(lip_neg_path)
    lip_gen = pd.read_pickle(lip_gen_path)
    lip_tot = pd.read_pickle(lip_tot_path)
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
        (lip_pos, _PANSS_LABELS["pos"], axes[0, 0], "A"),
        (lip_neg, _PANSS_LABELS["neg"], axes[0, 1], "B"),
        (lip_gen, _PANSS_LABELS["gen"], axes[1, 0], "C"),
        (lip_tot, _PANSS_LABELS["tot"], axes[1, 1], "D"),
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
        panel_leg = ax.get_legend()
        if panel_leg:
            if i == 0:
                leg_handles = panel_leg.legend_handles
                leg_labels = [t.get_text() for t in panel_leg.get_texts()]
            panel_leg.remove()

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

    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)


@task
def task_plot_panss_outcome_enrichment_combined(
    results_pos_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_sum_pos.pkl",
    results_neg_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_sum_neg.pkl",
    results_gen_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_sum_gen.pkl",
    results_tot_path=UNIVARIATE_LIPID_PANSS_DIR
    / "univariate_lipid_results_standard_panss_total_score.pkl",
    enrich_pos_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_sum_pos.pkl",
    enrich_neg_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_sum_neg.pkl",
    enrich_gen_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_gen.pkl",
    enrich_tot_path=UNIVARIATE_LIPID_PANSS_DIR
    / "lipid_enrichment_results_panss_total_score.pkl",
    annotation_df_path=ANNOTATION_DF_PATH,
    bld_plots_dir_output: Annotated[Path, Product] = BLD_ENRICHMENT_PLOT_DIR
    / "panss_outcome_enrichment_combined.svg",
):
    """Combined lipid enrichment (boxplot style) with each PANSS subscale as outcome.

    Layout: 2 rows × 2 columns (A4-friendly portrait).
      A (top-left)     PANSS Positive
      B (top-right)    PANSS Negative
      C (bottom-left)  PANSS General
      D (bottom-right) PANSS Total Score
      Shared legend centred at the bottom; shared y-label via supylabel.
    """
    results_pos = pd.read_pickle(results_pos_path)
    results_neg = pd.read_pickle(results_neg_path)
    results_gen = pd.read_pickle(results_gen_path)
    results_tot = pd.read_pickle(results_tot_path)
    enrich_pos = pd.read_pickle(enrich_pos_path)
    enrich_neg = pd.read_pickle(enrich_neg_path)
    enrich_gen = pd.read_pickle(enrich_gen_path)
    enrich_tot = pd.read_pickle(enrich_tot_path)
    annotation_df = pd.read_pickle(annotation_df_path)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.95,
        bottom=0.22,
        hspace=0.65,
        wspace=0.35,
    )

    models = [
        (results_pos, enrich_pos, _PANSS_LABELS["pos"], axes[0, 0], "A"),
        (results_neg, enrich_neg, _PANSS_LABELS["neg"], axes[0, 1], "B"),
        (results_gen, enrich_gen, _PANSS_LABELS["gen"], axes[1, 0], "C"),
        (results_tot, enrich_tot, _PANSS_LABELS["tot"], axes[1, 1], "D"),
    ]

    leg_handles, leg_labels = [], []
    for i, (results_df, enrich_df, title, ax, label) in enumerate(models):
        plot_lipid_coef_distributions(results_df, annotation_df, enrich_df, ax=ax)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel("")
        ax.text(
            -0.10,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        panel_leg = ax.get_legend()
        if panel_leg:
            if i == 0:
                leg_handles = panel_leg.legend_handles
                leg_labels = [t.get_text() for t in panel_leg.get_texts()]
            panel_leg.remove()

    fig.supylabel(
        "Coefficient (lipid × severe psychosis subtype probability)",
        fontsize=10,
        x=0.01,
    )

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

    fig.savefig(bld_plots_dir_output, bbox_inches="tight")
    plt.close(fig)
