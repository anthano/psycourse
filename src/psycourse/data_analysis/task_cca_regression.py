import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.cca_regression import cca_regression_analysis

RESULTS_DIR = BLD_RESULTS / "cca_regression"

prs_cols = [
    "BD_PRS",
    "SCZ_PRS",
    "MDD_PRS",
    "ADHD_PRS",
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


def task_cca_regression(
    product: Annotated[Path, Product] = RESULTS_DIR / "results.pkl",
) -> None:
    df = pd.read_pickle(BLD_DATA / "multimodal_lipid_subset_df.pkl")

    results_dict = cca_regression_analysis(
        df=df, prs_cols=prs_cols, lipid_class_cols=lipid_cols
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    with open(product, "wb") as f:
        pickle.dump(results_dict, f)
