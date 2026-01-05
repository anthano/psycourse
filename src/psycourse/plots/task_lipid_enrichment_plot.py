from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.plots.lipid_enrichment_plot import (
    enrichment_strength_plot,
    plot_lipid_coef_distributions,
)

ANNOT_DF_PATH = BLD_DATA / "cleaned_lipid_class_data.pkl"
LIPID_RESULTS_PATH = BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid"
PLOT_PATH = BLD_RESULTS / "plots" / "enrichment_analysis"

LIPID_PLOT_VARIANTS = {
    "default": {
        "results_df": LIPID_RESULTS_PATH / "univariate_lipid_results.pkl",
        "enrichment_df": LIPID_RESULTS_PATH / "lipid_enrichment_results.pkl",
        "strength_plot_path": PLOT_PATH / "lipid_enrichment_strength_plot.svg",
        "coef_plot_path": PLOT_PATH / "lipid_enrichment_bp_plot.svg",
    },
    "cov_diagnosis": {
        "results_df": LIPID_RESULTS_PATH / "univariate_lipid_results_cov_diagnosis.pkl",
        "enrichment_df": LIPID_RESULTS_PATH
        / "lipid_enrichment_results_cov_diagnosis.pkl",
        "strength_plot_path": PLOT_PATH
        / "lipid_enrichment_strength_plot_cov_diagnosis.svg",
        "coef_plot_path": PLOT_PATH / "lipid_enrichment_bp_plot_cov_diagnosis.svg",
    },
    "cov_med": {
        "results_df": LIPID_RESULTS_PATH / "univariate_lipid_results_cov_med.pkl",
        "enrichment_df": LIPID_RESULTS_PATH / "lipid_enrichment_results_cov_med.pkl",
        "strength_plot_path": PLOT_PATH / "lipid_enrichment_strength_plot_cov_med.svg",
        "coef_plot_path": PLOT_PATH / "lipid_enrichment_bp_plot_cov_med.svg",
    },
    "cov_med_and_diag": {
        "results_df": LIPID_RESULTS_PATH
        / "univariate_lipid_results_cov_med_and_diag.pkl",
        "enrichment_df": LIPID_RESULTS_PATH
        / "lipid_enrichment_results_cov_med_and_diag.pkl",
        "strength_plot_path": PLOT_PATH
        / "lipid_enrichment_strength_plot_cov_med_and_diag.svg",
        "coef_plot_path": PLOT_PATH / "lipid_enrichment_bp_plot_cov_med_and_diag.svg",
    },
}


for variant, paths in LIPID_PLOT_VARIANTS.items():

    @pytask.task(
        id=f"{variant}_strength_plot",
        kwargs={
            "variant": variant,
            "enrichment_results_df_path": paths["enrichment_df"],
            "produces": paths["strength_plot_path"],
        },
    )
    def task_lipid_enrichment_strength_plot(
        enrichment_results_df_path: Path,
        produces: Annotated[Path, pytask.Product],
        variant: str,
    ):
        results_df = pd.read_pickle(enrichment_results_df_path)
        plot_fig, _ = enrichment_strength_plot(results_df, variant=variant)
        plot_fig.savefig(produces)

    @pytask.task(
        id=f"{variant}_distribution_plot",
        kwargs={
            "variant": variant,
            "results_df_path": paths["results_df"],
            "annot_df_path": ANNOT_DF_PATH,
            "enrichment_df_path": paths["enrichment_df"],
            "produces": paths["coef_plot_path"],
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
        fig, _ = plot_lipid_coef_distributions(results_df, annot_df, enrich_df, variant)
        fig.savefig(produces)
