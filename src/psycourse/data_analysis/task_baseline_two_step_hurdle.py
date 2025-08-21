import itertools
from itertools import product
from pathlib import Path

import pandas as pd
from pytask import task

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.explorative_hurdle_wrappers import (
    explorative_run_hurdle_analysis,
)

DATA = BLD_DATA / "multimodal_complete_df.pkl"
BASELINE_BLD_HURDLE = BLD_RESULTS / "multivariate" / "hurdle_runs" / "baseline"


CUTOFFS = [0.25]
INNERS = [2]
OUTERS = [2]
N_REPEATS = 1

VIEWS = ["lipids_only", "prs_only"]

for view in VIEWS:
    for cutoff, inner, outer in itertools.product(CUTOFFS, INNERS, OUTERS):
        combination_id = f"{cutoff=}_{inner=}_{outer=}"

    BASELINE_HURDLE_ANALYSIS_RESULTS_PATHS: dict[str, Path] = {
        "metrics_df": BASELINE_BLD_HURDLE / f"metrics_df__{view}_{combination_id}.pkl"
    }

    @task(f"{combination_id}_{view}")
    def task_hurdle_repeat(
        depends_on: Path = BLD_DATA / "multimodal_complete_df.pkl",
        produces: dict[str, Path] = BASELINE_HURDLE_ANALYSIS_RESULTS_PATHS,
        view="multimodal",
        cutoff: float = cutoff,
        inner: int = inner,
        outer: int = outer,
    ):
        multimodal_complete_df = pd.read_pickle(depends_on)

        metrics_df = explorative_run_hurdle_analysis(
            multimodal_complete_df,
            view=view,
            cutoff=cutoff,
            inner=inner,
            outer=outer,
            n_repeats=N_REPEATS,
            base_seed=42,
            clf_n_jobs=1,
            reg_n_jobs=1,
        )

        metrics_df.to_pickle(produces["metrics_df"])

    # ---------------- aggregation per combination (putting all repeats together) ------
    summary_metrics_path: dict[str, Path] = {
        "metrics_summary_df": BASELINE_BLD_HURDLE
        / f"summary__{view}_{combination_id}.pkl",
    }

    @task(f"{combination_id}_{view}")
    def task_hurdle_aggregate(
        depends_on: dict[str, Path] = BASELINE_HURDLE_ANALYSIS_RESULTS_PATHS,
        produces: dict[str, Path] = summary_metrics_path,
        view: str = view,
        cutoff: float = cutoff,
        inner: int = inner,
        outer: int = outer,
    ):
        metrics_df = pd.read_pickle(depends_on["metrics_df"])

        summary = {
            "combo_id": f"{cutoff}_{inner}_{outer}",
            "view": view,
            "cutoff_quantile": float(cutoff),
            "n_inner_cv": int(inner),
            "n_outer_cv": int(outer),
            "n_repeats": int(len(metrics_df)),
        }

        for col in [
            "test_accuracy",
            "test_balanced_accuracy",
            "test_avg_precision",
            "test_mcc",
            "test_prevalence",
            "test_roc_auc",
            "test_precision",
            "test_recall",
            "test_r2",
            "test_mse",
            "test_regression_mae",
            "test_regression_rmse",
        ]:
            summary[f"{col}_mean"] = metrics_df[col].mean()
            summary[f"{col}_std"] = metrics_df[col].std()

        pd.DataFrame([summary]).to_pickle(produces["metrics_summary_df"])

## ---------------- final collection across all combinations ----------------

for view in VIEWS:
    FINAL_SUMMARY_DEPENDS_ON = [
        BASELINE_BLD_HURDLE / f"summary__{view}_{cutoff=}_{inner=}_{outer=}.pkl"
        for cutoff, inner, outer in product(CUTOFFS, INNERS, OUTERS)
    ]

    @task(f"final_summary_{view}")
    def task_final_collection(
        depends_on: list[Path] = FINAL_SUMMARY_DEPENDS_ON,
        produces: Path = BASELINE_BLD_HURDLE / f"{view}_baseline_summary.pkl",
    ):
        summary_dfs = [pd.read_pickle(path) for path in depends_on]
        final_summary_df = pd.concat(summary_dfs, ignore_index=True)
        final_summary_df.to_pickle(produces)
