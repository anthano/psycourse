from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import task

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.lipid_permutation_analysis import (
    lipid_class_enrichment_gsea,
)

REGRESSION_RESULTS_PATH = BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid"

LIPID_ENRICHMENT_INPUTS = {
    "default": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results.pkl",
        "produces": REGRESSION_RESULTS_PATH / "lipid_enrichment_results.pkl",
    },
    "cov_diagnosis": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results_cov_diagnosis.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "lipid_enrichment_results_cov_diagnosis.pkl",
    },
    "cov_med": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results_cov_med.pkl",
        "produces": REGRESSION_RESULTS_PATH / "lipid_enrichment_results_cov_med.pkl",
    },
    "cov_med_and_diag": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results_cov_med_and_diag.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "lipid_enrichment_results_cov_med_and_diag.pkl",
    },
}


for variant, paths in LIPID_ENRICHMENT_INPUTS.items():

    @task(id=f"lipid_enrichment_{variant}")
    def task_lipid_enrichment_analysis(
        regression_results_df_path=paths["regression_results_df_path"],
        annot_df_path=BLD_DATA / "cleaned_lipid_class_data.pkl",
        produces: Annotated[Path, pytask.Product] = paths["produces"],
    ):
        results_df = pd.read_pickle(regression_results_df_path)
        annot_df = pd.read_pickle(annot_df_path)

        enrichment_results = lipid_class_enrichment_gsea(results_df, annot_df)
        enrichment_results.to_pickle(produces)
