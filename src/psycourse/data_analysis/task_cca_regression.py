import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.cca_regression import cca_regression_analysis
from psycourse.data_analysis.univariate_analysis import main_prs_columns

RESULTS_DIR = BLD_RESULTS / "cca_regression"

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
    data_path: Path = BLD_DATA / "multimodal_lipid_subset_df.pkl",
    product: Annotated[Path, Product] = RESULTS_DIR / "results.pkl",
) -> None:
    df = pd.read_pickle(data_path)

    results_dict = cca_regression_analysis(
        df=df, prs_cols=main_prs_columns(df), lipid_class_cols=lipid_cols
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    with open(product, "wb") as f:
        pickle.dump(results_dict, f)
