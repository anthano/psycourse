from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import pytask

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.plots.lipid_enrichment_plot import (
    enrichment_strength_plot,
    plot_lipid_coef_distributions,
)

ANNOT_DF_PATH = BLD_DATA / "cleaned_lipid_class_data.pkl"
LIPID_RESULTS_PATH = BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid"
MED_ADJ_RESULTS_PATH = LIPID_RESULTS_PATH / "medication_adjusted"
PLOT_PATH = BLD_RESULTS / "plots" / "enrichment_analysis"


LIPID_PLOT_VARIANTS = {
    "default": {
        "results_df": "univariate_lipid_results.pkl",
        "enrichment_df": "lipid_enrichment_results.pkl",
    },
    "cov_diagnosis": {
        "results_df": "univariate_lipid_results_cov_diagnosis.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_diagnosis.pkl",
    },
    "cov_med": {
        "results_df": "univariate_lipid_results_cov_med.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_med.pkl",
    },
    "cov_med_and_diag": {
        "results_df": "univariate_lipid_results_cov_med_and_diag.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_med_and_diag.pkl",
    },
    "cov_panss": {
        "results_df": "univariate_lipid_results_cov_panss.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_panss.pkl",
    },
}


for variant, files in LIPID_PLOT_VARIANTS.items():
    enrichment_df_path = LIPID_RESULTS_PATH / files["enrichment_df"]
    results_df_path = LIPID_RESULTS_PATH / files["results_df"]

    # STRENGTH PLOT
    @pytask.task(
        id=f"{variant}_strength_plot",
        kwargs={
            "variant": variant,
            "enrichment_results_df_path": enrichment_df_path,
            "produces": PLOT_PATH
            / (
                f"lipid_enrichment_strength_plot_{variant}.svg"
                if variant != "default"
                else "lipid_enrichment_strength_plot.svg"
            ),
        },
    )
    def task_lipid_enrichment_strength_plot(
        enrichment_results_df_path: Path,
        produces: Annotated[Path, pytask.Product],
        variant: str,
    ):
        df = pd.read_pickle(enrichment_results_df_path)
        fig, _ = enrichment_strength_plot(df, variant=variant)
        fig.savefig(produces)

    # COEFFICIENT DISTRIBUTION PLOT
    @pytask.task(
        id=f"{variant}_distribution_plot",
        kwargs={
            "variant": variant,
            "results_df_path": results_df_path,
            "annot_df_path": ANNOT_DF_PATH,
            "enrichment_df_path": enrichment_df_path,
            "produces": PLOT_PATH
            / (
                f"lipid_enrichment_bp_plot_{variant}.svg"
                if variant != "default"
                else "lipid_enrichment_bp_plot.svg"
            ),
        },
    )
    def task_lipid_coef_distribution_plot(
        results_df_path: Path,
        annot_df_path: Path,
        enrichment_df_path: Path,
        produces: Annotated[Path, pytask.Product],
        variant: str,
    ):
        results_df = pd.read_pickle(results_df_path)
        annot_df = pd.read_pickle(annot_df_path)
        enrich_df = pd.read_pickle(enrichment_df_path)
        fig, _ = plot_lipid_coef_distributions(results_df, annot_df, enrich_df)
        fig.savefig(produces, bbox_inches="tight")


COMBINED_VARIANTS = [
    ("default", "A", "Primary Model"),
    ("cov_diagnosis", "B", "Covariate: diagnosis"),
    ("cov_med", "C", "Covariate: medication"),
    ("cov_med_and_diag", "D", "Covariate: medication + diagnosis"),
]


@pytask.task(
    id="combined_bp_plot",
    kwargs={
        "annot_df_path": ANNOT_DF_PATH,
        "results_df_paths": [
            LIPID_RESULTS_PATH / LIPID_PLOT_VARIANTS[v]["results_df"]
            for v, _, _ in COMBINED_VARIANTS
        ],
        "enrichment_df_paths": [
            LIPID_RESULTS_PATH / LIPID_PLOT_VARIANTS[v]["enrichment_df"]
            for v, _, _ in COMBINED_VARIANTS
        ],
        "produces": PLOT_PATH / "lipid_enrichment_bp_plot_combined.svg",
    },
)
def task_lipid_coef_distribution_combined(
    annot_df_path: Path,
    results_df_paths: list[Path],
    enrichment_df_paths: list[Path],
    produces: Annotated[Path, pytask.Product],
):
    annot_df = pd.read_pickle(annot_df_path)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(9, 12),
        constrained_layout=False,  # ← changed
    )

    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.05, hspace=0.5)

    for ax, (_variant, panel_label, covariate_label), results_path, enrich_path in zip(
        axes, COMBINED_VARIANTS, results_df_paths, enrichment_df_paths, strict=False
    ):
        results_df = pd.read_pickle(results_path)
        enrich_df = pd.read_pickle(enrich_path)

        plot_lipid_coef_distributions(results_df, annot_df, enrich_df, ax=ax)
        ax.set_ylabel("")  # ← strip individual labels

        ax.text(
            -0.08,
            1.18,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_title(covariate_label, loc="left", fontsize=11, pad=4)

    fig.supylabel(
        "Coefficient (lipid X severe psychosis subtype probability)",
        fontsize=10,
        x=0.02,
    )

    fig.savefig(produces, bbox_inches="tight")
    plt.close()


# ======================================================================================
# MEDICATION-ADJUSTED ENRICHMENT PLOTS (one medication class at a time)
# ======================================================================================

LIPID_PLOT_VARIANTS_MED_ADJ = {
    "cov_antidepressants": {
        "results_df": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_antidepressants.pkl",
        "enrichment_df": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_antidepressants.pkl",
    },
    "cov_antipsychotics": {
        "results_df": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_antipsychotics.pkl",
        "enrichment_df": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_antipsychotics.pkl",
    },
    "cov_tranquilizers": {
        "results_df": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_tranquilizers.pkl",
        "enrichment_df": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_tranquilizers.pkl",
    },
    "cov_mood_stabilizers": {
        "results_df": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_mood_stabilizers.pkl",
        "enrichment_df": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_mood_stabilizers.pkl",
    },
}

for variant, files in LIPID_PLOT_VARIANTS_MED_ADJ.items():
    enrichment_df_path = files["enrichment_df"]
    results_df_path = files["results_df"]

    @pytask.task(
        id=f"{variant}_strength_plot",
        kwargs={
            "variant": variant,
            "enrichment_results_df_path": enrichment_df_path,
            "produces": PLOT_PATH / f"lipid_enrichment_strength_plot_{variant}.svg",
        },
    )
    def task_lipid_enrichment_strength_plot_med_adj(
        enrichment_results_df_path: Path,
        produces: Annotated[Path, pytask.Product],
        variant: str,
    ):
        df = pd.read_pickle(enrichment_results_df_path)
        fig, _ = enrichment_strength_plot(df, variant=variant)
        fig.savefig(produces)

    @pytask.task(
        id=f"{variant}_distribution_plot",
        kwargs={
            "variant": variant,
            "results_df_path": results_df_path,
            "annot_df_path": ANNOT_DF_PATH,
            "enrichment_df_path": enrichment_df_path,
            "produces": PLOT_PATH / f"lipid_enrichment_bp_plot_{variant}.svg",
        },
    )
    def task_lipid_coef_distribution_plot_med_adj(
        results_df_path: Path,
        annot_df_path: Path,
        enrichment_df_path: Path,
        produces: Annotated[Path, pytask.Product],
        variant: str,
    ):
        results_df = pd.read_pickle(results_df_path)
        annot_df = pd.read_pickle(annot_df_path)
        enrich_df = pd.read_pickle(enrichment_df_path)
        fig, _ = plot_lipid_coef_distributions(results_df, annot_df, enrich_df)
        fig.savefig(produces, bbox_inches="tight")


COMBINED_VARIANTS_MED_ADJ = [
    ("cov_antidepressants", "A", "Covariate: antidepressants"),
    ("cov_antipsychotics", "B", "Covariate: antipsychotics"),
    ("cov_tranquilizers", "C", "Covariate: tranquilizers"),
    ("cov_mood_stabilizers", "D", "Covariate: mood stabilizers"),
]


@pytask.task(
    id="combined_bp_plot_med_adj",
    kwargs={
        "annot_df_path": ANNOT_DF_PATH,
        "results_df_paths": [
            LIPID_PLOT_VARIANTS_MED_ADJ[v]["results_df"]
            for v, _, _ in COMBINED_VARIANTS_MED_ADJ
        ],
        "enrichment_df_paths": [
            LIPID_PLOT_VARIANTS_MED_ADJ[v]["enrichment_df"]
            for v, _, _ in COMBINED_VARIANTS_MED_ADJ
        ],
        "produces": PLOT_PATH / "lipid_enrichment_bp_plot_combined_med_adj.svg",
    },
)
def task_lipid_coef_distribution_combined_med_adj(
    annot_df_path: Path,
    results_df_paths: list[Path],
    enrichment_df_paths: list[Path],
    produces: Annotated[Path, pytask.Product],
):
    annot_df = pd.read_pickle(annot_df_path)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(9, 12),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.05, hspace=0.5)

    for ax, (_variant, panel_label, covariate_label), results_path, enrich_path in zip(
        axes,
        COMBINED_VARIANTS_MED_ADJ,
        results_df_paths,
        enrichment_df_paths,
        strict=False,
    ):
        results_df = pd.read_pickle(results_path)
        enrich_df = pd.read_pickle(enrich_path)

        plot_lipid_coef_distributions(results_df, annot_df, enrich_df, ax=ax)
        ax.set_ylabel("")

        ax.text(
            -0.08,
            1.18,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_title(covariate_label, loc="left", fontsize=11, pad=4)

    fig.supylabel(
        "Coefficient (lipid X severe psychosis subtype probability)",
        fontsize=10,
        x=0.02,
    )

    fig.savefig(produces, bbox_inches="tight")
    plt.close()


# ======================================================================================

LIPID_PLOT_VARIANTS_COV_PANSS = {
    "cov_panss_pos": {
        "results_df": "univariate_lipid_results_cov_panss.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_panss.pkl",
    },
    "cov_panss_neg": {
        "results_df": "univariate_lipid_results_cov_panss_neg.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_panss_neg.pkl",
    },
    "cov_panss_gen": {
        "results_df": "univariate_lipid_results_cov_panss_gen.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_panss_gen.pkl",
    },
    "cov_panss_total": {
        "results_df": "univariate_lipid_results_cov_panss_total_score.pkl",
        "enrichment_df": "lipid_enrichment_results_cov_panss_total_score.pkl",
    },
}

COMBINED_VARIANTS_COV_PANSS = [
    ("cov_panss_pos", "A", "Covariate: PANSS positive symptoms"),
    ("cov_panss_neg", "B", "Covariate: PANSS negative symptoms"),
    ("cov_panss_gen", "C", "Covariate: PANSS general psychopathology scale"),
    ("cov_panss_total", "D", "Covariate: PANSS total scale"),
]


@pytask.task(
    id="combined_bp_plot_cov_panss",
    kwargs={
        "annot_df_path": ANNOT_DF_PATH,
        "results_df_paths": [
            LIPID_RESULTS_PATH / LIPID_PLOT_VARIANTS_COV_PANSS[v]["results_df"]
            for v, _, _ in COMBINED_VARIANTS_COV_PANSS
        ],
        "enrichment_df_paths": [
            LIPID_RESULTS_PATH / LIPID_PLOT_VARIANTS_COV_PANSS[v]["enrichment_df"]
            for v, _, _ in COMBINED_VARIANTS_COV_PANSS
        ],
        "produces": PLOT_PATH / "lipid_enrichment_bp_plot_combined_cov_panss.svg",
    },
)
def task_lipid_coef_distribution_combined_cov_panss(
    annot_df_path: Path,
    results_df_paths: list[Path],
    enrichment_df_paths: list[Path],
    produces: Annotated[Path, pytask.Product],
):
    annot_df = pd.read_pickle(annot_df_path)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(11.69, 8.27),
        constrained_layout=False,
    )

    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.05, hspace=0.5)

    for ax, (_variant, panel_label, covariate_label), results_path, enrich_path in zip(
        axes,
        COMBINED_VARIANTS_COV_PANSS,
        results_df_paths,
        enrichment_df_paths,
        strict=False,
    ):
        results_df = pd.read_pickle(results_path)
        enrich_df = pd.read_pickle(enrich_path)

        plot_lipid_coef_distributions(results_df, annot_df, enrich_df, ax=ax)
        ax.set_ylabel("")  # ← strip individual labels

        ax.text(
            -0.08,
            1.18,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_title(covariate_label, loc="left", fontsize=11, pad=4)

    fig.supylabel(
        "Coefficient (lipid X severe psychosis subtype probability)",
        fontsize=10,
        x=0.02,
    )

    fig.savefig(produces, bbox_inches="tight")
    plt.close()
