from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask

from psycourse.config import BLD_DATA, BLD_RESULTS, WRITING
from psycourse.plots.lipid_enrichment_plot import (
    enrichment_strength_plot,
    plot_lipid_coef_distributions,
)

ANNOT_DF_PATH = BLD_DATA / "cleaned_lipid_class_data.pkl"
LIPID_RESULTS_PATH = BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid"
PLOT_PATH = BLD_RESULTS / "plots" / "enrichment_analysis"
WRITING_PLOT_PATH = WRITING / "plots" / "enrichment_analysis"


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
}


def dual_output(name: str) -> list[Path]:
    return [PLOT_PATH / name, WRITING_PLOT_PATH / name]


for variant, files in LIPID_PLOT_VARIANTS.items():
    enrichment_df_path = LIPID_RESULTS_PATH / files["enrichment_df"]
    results_df_path = LIPID_RESULTS_PATH / files["results_df"]

    # STRENGTH PLOT
    @pytask.task(
        id=f"{variant}_strength_plot",
        kwargs={
            "variant": variant,
            "enrichment_results_df_path": enrichment_df_path,
            "produces": dual_output(f"lipid_enrichment_strength_plot_{variant}.svg")
            if variant != "default"
            else dual_output("lipid_enrichment_strength_plot.svg"),
        },
    )
    def task_lipid_enrichment_strength_plot(
        enrichment_results_df_path: Path,
        produces: Annotated[list[Path], pytask.Product],
        variant: str,
    ):
        df = pd.read_pickle(enrichment_results_df_path)
        fig, _ = enrichment_strength_plot(df, variant=variant)
        for path in produces:
            fig.savefig(path)

    # COEFFICIENT DISTRIBUTION PLOT
    @pytask.task(
        id=f"{variant}_distribution_plot",
        kwargs={
            "variant": variant,
            "results_df_path": results_df_path,
            "annot_df_path": ANNOT_DF_PATH,
            "enrichment_df_path": enrichment_df_path,
            "produces": dual_output(f"lipid_enrichment_bp_plot_{variant}.svg")
            if variant != "default"
            else dual_output("lipid_enrichment_bp_plot.svg"),
        },
    )
    def task_lipid_coef_distribution_plot(
        results_df_path: Path,
        annot_df_path: Path,
        enrichment_df_path: Path,
        produces: Annotated[list[Path], pytask.Product],
        variant: str,
    ):
        results_df = pd.read_pickle(results_df_path)
        annot_df = pd.read_pickle(annot_df_path)
        enrich_df = pd.read_pickle(enrichment_df_path)
        fig, _ = plot_lipid_coef_distributions(results_df, annot_df, enrich_df)
        for path in produces:
            fig.savefig(path, bbox_inches="tight")
