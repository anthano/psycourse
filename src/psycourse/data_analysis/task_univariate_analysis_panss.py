from __future__ import annotations

from pathlib import Path
from typing import Annotated, Callable

import pandas as pd
import pytask
from pytask import Product

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_analysis_panss import (
    univariate_lipid_regression_cov_diagnosis_panss,
    univariate_lipid_regression_cov_med_and_diag_panss,
    univariate_lipid_regression_cov_med_panss,
    univariate_lipid_regression_panss,
    univariate_prs_regression_panss,
)

UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "prs" / "panss"
)
UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid" / "panss"
)

PANSS_COLUMNS = (
    "panss_sum_pos",
    "panss_sum_neg",
    "panss_sum_gen",
    "panss_total_score",
)

SCRIPT_PATH = SRC / "data_analysis" / "univariate_analysis_panss.py"
MULTIMODAL_DF_PATH = BLD_DATA / "multimodal_complete_df.pkl"


def _lipid_products(kind: str, col: str) -> dict[str, Path]:
    return {
        "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
        / f"n_subset_dict_{kind}_{col}.pkl",
        "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
        / f"univariate_lipid_results_top20_{kind}_{col}.pkl",
        "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
        / f"univariate_lipid_results_{kind}_{col}.pkl",
    }


# ======================================================================================
# PRS TASKS (one task per PANSS col)
# ======================================================================================

for col in PANSS_COLUMNS:
    _prs_output = (
        UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
        / f"univariate_prs_results_panss_standard_cov_{col}.pkl"
    )

    @pytask.task(
        id=f"prs-standard-{col}",
        kwargs={
            "col": col,
            "script_path": SCRIPT_PATH,
            "multimodal_df_path": MULTIMODAL_DF_PATH,
            "produces": _prs_output,
        },
    )
    def task_univariate_prs_panss_one(
        col: str,
        script_path: Path,
        multimodal_df_path: Path,
        produces: Annotated[Path, Product],
    ) -> None:
        data = pd.read_pickle(multimodal_df_path)
        _n_subset_dict, results_df = univariate_prs_regression_panss(data, col)
        results_df.to_pickle(produces)


# ======================================================================================
# LIPID TASKS (one task per PANSS col)
# ======================================================================================

_LIPID_SPECS: dict[str, Callable] = {
    "standard": univariate_lipid_regression_panss,
    "cov_diagnosis": univariate_lipid_regression_cov_diagnosis_panss,
    "cov_med": univariate_lipid_regression_cov_med_panss,
    "cov_med_and_diag": univariate_lipid_regression_cov_med_and_diag_panss,
}

for kind, func in _LIPID_SPECS.items():
    for col in PANSS_COLUMNS:
        products = _lipid_products(kind=kind, col=col)

        @pytask.task(
            id=f"lipid-{kind}-{col}",
            kwargs={
                "col": col,
                "func": func,
                "script_path": SCRIPT_PATH,
                "multimodal_df_path": MULTIMODAL_DF_PATH,
                "n_subset_dict_path": products["n_subset_dict"],
                "top20_lipids_path": products["top20_lipids"],
                "univariate_lipid_results_path": products["univariate_lipid_results"],
            },
        )
        def task_univariate_lipid_one(
            col: str,
            func: Callable,
            script_path: Path,  # tracked dependency
            multimodal_df_path: Path,  # tracked dependency
            n_subset_dict_path: Annotated[Path, Product],
            top20_lipids_path: Annotated[Path, Product],
            univariate_lipid_results_path: Annotated[Path, Product],
        ) -> None:
            data = pd.read_pickle(multimodal_df_path)
            n_subset_dict, top20_lipids, univariate_lipid_results = func(data, col)

            pd.to_pickle(n_subset_dict, n_subset_dict_path)
            top20_lipids.to_pickle(top20_lipids_path)
            univariate_lipid_results.to_pickle(univariate_lipid_results_path)
