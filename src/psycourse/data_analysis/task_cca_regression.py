from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.cca_regression import cca_prs_lipids_regression

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
    "class_LPE",
    "class_PC",
    "class_PC_O",
    "class_PC_P",
    "class_PE",
    "class_PE_P",
    "class_TAG",
    "class_dCer",
    "class_dSM",
    "class_CAR",
    "class_CE",
    "class_DAG",
    "class_FA",
    "class_LPC",
    "class_LPC_O",
    "class_LPC_P",
]

score_cols = ["PRS_CCA_Component_1", "Lipid_CCA_Component_1"]

for score_col in score_cols:
    cca_result_path = RESULTS_DIR / f"cca_{score_col}_results.pkl"
    regression_result_path = RESULTS_DIR / f"cca_{score_col}_regression_results.pkl"

    @task(id=score_col)
    def _(  # noqa: F811
        score_col: str[score_col],
        produces_cca_result: Annotated[Path, Product] = cca_result_path,
        produces_regression_result: Annotated[Path, Product] = regression_result_path,
    ) -> None:
        df = pd.read_pickle(BLD_DATA / "multimodal_lipid_subset_df.pkl")
        cca_result_df, result_df = cca_prs_lipids_regression(
            multimodal_lipid_subset_df=df,
            prs_cols=prs_cols,
            lipid_cols=lipid_cols,
            score_col=score_col,
            n_permutations=10000,
            random_state=42,
        )
        cca_result_df.to_pickle(produces_cca_result)
        result_df.to_pickle(produces_regression_result)
