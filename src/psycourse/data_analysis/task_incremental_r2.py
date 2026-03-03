import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.incremental_r2 import (
    incremental_r2_decomposition,
    make_r2_table,
)

RESULTS_DIR = BLD_RESULTS / "incremental_r2"

prs_cols = [
    "BD_PRS",
    "SCZ_PRS",
    "MDD_PRS",
    "ADHD_PRS",
    "ASD_PRS",
    "Alzheimer_PRS",
    "Education_PRS",
    "Agreeableness_PRS",
    "Conscientiousness_PRS",
    "Extraversion_PRS",
    "Neuroticism_PRS",
    "Openness_PRS",
    "SleepDurationLong_PRS",
    "SleepDurationShort_PRS",
]

lipid_cols = [
    "LPE",
    "PC",
    "PC_O",
    "PC_P",
    "PE",
    "PE_P",
    "TAG",
    "dCer",
    "dSM",
    "CAR",
    "CE",
    "DAG",
    "FA",
    "LPC",
    "LPC_O",
    "LPC_P",
]


def task_incremental_r2_decomposition(
    results_path: Annotated[Path, Product] = RESULTS_DIR / "incremental_r2_results.pkl",
    table_path: Annotated[Path, Product] = RESULTS_DIR / "incremental_r2_table.pkl",
) -> None:
    df = pd.read_pickle(BLD_DATA / "multimodal_lipid_subset_df.pkl")

    results = incremental_r2_decomposition(
        df=df,
        prs_cols=prs_cols,
        lipid_cols=lipid_cols,
        outcome_col="prob_class_5",
        n_permutations=20_000,
        random_state=42,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    table = make_r2_table(results)
    table.to_pickle(table_path)
