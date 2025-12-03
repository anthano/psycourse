from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import task

from psycourse.config import BLD_RESULTS
from psycourse.plots.lipid_enrichment_plot import enrichment_strength_plot

LIPID_ENRICHMENT_INPUTS = {
    "default": {
        "enrichment_results": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "lipid"
        / "lipid_enrichment_results.pkl",
        "produces": BLD_RESULTS / "plots" / "lipid_enrichment_plot.svg",
    },
    "cov_diag": {
        "enrichment_results": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "lipid"
        / "lipid_enrichment_results_cov_diag.pkl",
        "produces": BLD_RESULTS / "plots" / "lipid_enrichment_plot_cov_diag.svg",
    },
}


for variant, paths in LIPID_ENRICHMENT_INPUTS.items():

    @task(id=f"lipid_enrichment_{variant}")
    def task_lipid_enrichment_strength_plot(
        enrichment_results_df_path=paths["enrichment_results"],
        produces: Annotated[Path, pytask.Product] = paths["produces"],
    ):
        results_df = pd.read_pickle(enrichment_results_df_path)

        plot_fig, plot_ax = enrichment_strength_plot(results_df, variant=variant)  # noqa:B023
        plot_fig.savefig(produces)
