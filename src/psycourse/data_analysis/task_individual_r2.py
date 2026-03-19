from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.incremental_r2 import individual_predictor_r2_table

RESULTS_DIR = BLD_RESULTS / "incremental_r2"

########################################################################################
# Predictors
########################################################################################

# The three PRS that were significant in the main incremental R² analysis
SIGNIFICANT_PRS_COLS = [
    "SCZ_PRS",
    "BD_PRS",
    "Education_PRS",  # EA-PRS
]

# The eight enriched lipid class scores — update this list with the actual
# enriched classes identified in the enrichment analysis.
ENRICHED_LIPID_COLS = [
    "LPE",
    "PC",
    "PC_O",
    "PC_P",
    "PE",
    "PE_P",
    "TAG",
    "dCer",
]


########################################################################################
# Task
########################################################################################


def task_individual_predictor_r2(
    data_path: Path = BLD_DATA / "multimodal_lipid_subset_df.pkl",
    results_path: Annotated[Path, Product] = RESULTS_DIR
    / "individual_predictor_r2.pkl",
    csv_path: Annotated[Path, Product] = RESULTS_DIR / "individual_predictor_r2.csv",
) -> None:
    """
    Compute individual ΔR² and permutation p-values (20 000 permutations) for each
    of the three significant PRS predictors (SCZ-PRS, BD-PRS, EA-PRS) and each of
    the eight enriched lipid class scores.

    Uses the same lipid-subset sample and block-specific residualization approach
    as the main incremental R² decomposition.

    Produces:
        individual_predictor_r2.pkl — DataFrame with columns:
            predictor, type, dR2, p_permutation
        individual_predictor_r2.csv — same table in CSV format
    """
    df = pd.read_pickle(data_path)

    results_table = individual_predictor_r2_table(
        df=df,
        prs_predictors=SIGNIFICANT_PRS_COLS,
        lipid_predictors=ENRICHED_LIPID_COLS,
        outcome_col="prob_class_5",
        n_permutations=20_000,
        random_state=42,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_table.to_pickle(results_path)
    results_table.to_csv(csv_path, index=False)
