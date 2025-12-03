from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask

from psycourse.config import BLD_RESULTS
from psycourse.plots.lipid_enrichment_plot import (
    enrichment_strength_plot,
)

LIPID_ENRICHMENT_PATHS = {
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


for variant, paths in LIPID_ENRICHMENT_PATHS.items():

    @pytask.task(
        id=variant,
        kwargs={
            "variant": variant,
            "enrichment_results_df_path": paths["enrichment_results"],
            "produces": paths["produces"],
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
