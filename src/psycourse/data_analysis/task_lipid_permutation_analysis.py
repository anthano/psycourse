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
MED_ADJ_RESULTS_PATH = REGRESSION_RESULTS_PATH / "medication_adjusted"

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
    "cov_panss": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results_cov_panss.pkl",
        "produces": REGRESSION_RESULTS_PATH / "lipid_enrichment_results_cov_panss.pkl",
    },
    "cov_panss_neg": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results_cov_panss_neg.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "lipid_enrichment_results_cov_panss_neg.pkl",
    },
    "cov_panss_gen": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results_cov_panss_gen.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "lipid_enrichment_results_cov_panss_gen.pkl",
    },
    "cov_panss_total_score": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "univariate_lipid_results_cov_panss_total_score.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "lipid_enrichment_results_cov_panss_total_score.pkl",
    },
    "panss_pos": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "panss"
        / "univariate_lipid_results_standard_panss_sum_pos.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "panss"
        / "lipid_enrichment_results_panss_sum_pos.pkl",
    },
    "panss_pos_med": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "panss"
        / "univariate_lipid_results_cov_med_panss_sum_pos.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "panss"
        / "lipid_enrichment_results_cov_med_panss_sum_pos.pkl",
    },
    "panss_neg": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "panss"
        / "univariate_lipid_results_standard_panss_sum_neg.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "panss"
        / "lipid_enrichment_results_panss_sum_neg.pkl",
    },
    "panss_neg_cov_med": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "panss"
        / "univariate_lipid_results_cov_med_panss_sum_neg.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "panss"
        / "lipid_enrichment_results_cov_med_panss_sum_neg.pkl",
    },
    "panss_gen": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "panss"
        / "univariate_lipid_results_standard_panss_sum_gen.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "panss"
        / "lipid_enrichment_results_panss_gen.pkl",
    },
    "panss_total_score": {
        "regression_results_df_path": REGRESSION_RESULTS_PATH
        / "panss"
        / "univariate_lipid_results_standard_panss_total_score.pkl",
        "produces": REGRESSION_RESULTS_PATH
        / "panss"
        / "lipid_enrichment_results_panss_total_score.pkl",
    },
    "cov_antidepressants": {
        "regression_results_df_path": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_antidepressants.pkl",
        "produces": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_antidepressants.pkl",
    },
    "cov_antipsychotics": {
        "regression_results_df_path": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_antipsychotics.pkl",
        "produces": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_antipsychotics.pkl",
    },
    "cov_tranquilizers": {
        "regression_results_df_path": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_tranquilizers.pkl",
        "produces": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_tranquilizers.pkl",
    },
    "cov_mood_stabilizers": {
        "regression_results_df_path": MED_ADJ_RESULTS_PATH
        / "univariate_lipid_results_cov_mood_stabilizers.pkl",
        "produces": MED_ADJ_RESULTS_PATH
        / "lipid_enrichment_results_cov_mood_stabilizers.pkl",
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
