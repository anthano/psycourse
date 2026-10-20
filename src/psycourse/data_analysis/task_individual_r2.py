from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.incremental_r2 import (
    individual_predictor_r2_table,
    significant_predictors,
)

RESULTS_DIR = BLD_RESULTS / "incremental_r2"

# Predictors are read from the primary results tables (FDR < 0.05) rather than
# listed by hand, so they stay in sync when upstream results change.
PRS_RESULTS_PATH = (
    BLD_RESULTS
    / "univariate"
    / "continuous_analysis"
    / "prs"
    / "univariate_prs_results_standard_cov.pkl"
)
LIPID_ENRICHMENT_RESULTS_PATH = (
    BLD_RESULTS
    / "univariate"
    / "continuous_analysis"
    / "lipid"
    / "lipid_enrichment_results.pkl"
)


########################################################################################
# Task
########################################################################################


def task_individual_predictor_r2(
    data_path: Path = BLD_DATA / "multimodal_lipid_subset_df.pkl",
    prs_results_path: Path = PRS_RESULTS_PATH,
    lipid_enrichment_results_path: Path = LIPID_ENRICHMENT_RESULTS_PATH,
    results_path: Annotated[Path, Product] = RESULTS_DIR
    / "individual_predictor_r2.pkl",
    csv_path: Annotated[Path, Product] = RESULTS_DIR / "individual_predictor_r2.csv",
) -> None:
    """
    Compute individual ΔR² and permutation p-values (20 000 permutations) for each
    significant PRS (primary PRS regression, FDR < 0.05) and each enriched lipid
    class score (primary lipid class enrichment, FDR < 0.05).

    Uses the same lipid-subset sample and block-specific residualization approach
    as the main incremental R² decomposition.

    Produces:
        individual_predictor_r2.pkl — DataFrame with columns:
            predictor, type, dR2, p_permutation
        individual_predictor_r2.csv — same table in CSV format
    """
    df = pd.read_pickle(data_path)
    prs_results = pd.read_pickle(prs_results_path)
    lipid_enrichment_results = pd.read_pickle(lipid_enrichment_results_path)

    results_table = individual_predictor_r2_table(
        df=df,
        prs_predictors=significant_predictors(prs_results),
        lipid_predictors=significant_predictors(lipid_enrichment_results),
        outcome_col="prob_class_5",
        n_permutations=20_000,
        random_state=42,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_table.to_pickle(results_path)
    results_table.to_csv(csv_path, index=False)
