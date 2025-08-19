import hashlib

import pandas as pd

from psycourse.data_analysis.explorative_two_step_hurdle import (
    explorative_stage_one_classification,
    explorative_stage_two_regression,
)


def explorative_run_hurdle_analysis(
    df, cutoff, inner, outer, n_repeats, base_seed=42, clf_n_jobs=1, reg_n_jobs=1
):
    """Run two-step hurdle with specified parameters for n repeats.
    Args:
        df(pd.DataFrame): The analysis data containing target and features.
        cutoff(float): The quantile cutoff for the first stage classification.
        inner(int): The number of inner cross-validation folds.
        outer(int): The number of outer cross-validation folds.
        n_repeats(int): How many times to repeat the run for this combination.
        base_seed(int): The base seed for reproducibility.
        clf_n_jobs(int): Number of jobs to run in parallel for classification.
        reg_n_jobs(int): Number of jobs to run in parallel for regression.
    Returns:
        tuple: A tuple containing three DataFrames:
            - metrics_df: DataFrame with metrics for each repeat.
            - clf_top20_df: DataFrame with top 20 features from classification
              for each repeat.
            - reg_top20_df: DataFrame with top 20 features from regression
              for each repeat.
    """
    metrics_data = {}

    for repeat in range(n_repeats):
        metrics_dict, clf_report, reg_report = run_single_combo(
            df,
            cutoff,
            inner,
            outer,
            repeat,
            base_seed=base_seed,
            clf_n_jobs=clf_n_jobs,
            reg_n_jobs=reg_n_jobs,
        )
        metrics_data[repeat] = metrics_dict

    metrics_df = pd.DataFrame.from_dict(metrics_data, orient="index")

    print(metrics_df.head())

    return metrics_df


def run_single_combo(
    df, cutoff, inner, outer, repeat, base_seed=42, clf_n_jobs=1, reg_n_jobs=1
) -> dict[str, float]:
    """Run a single combination of parameters for the two-step hurdle model.
    Args:
        df(pd.DataFrame): The analysis data containing target and features.
        cutoff(float): The quantile cutoff for the first stage classification.
        inner(int): The number of inner cross-validation folds.
        outer(int): The number of outer cross-validation folds.
        repeat(int): How many times to repeat the run for this combination.
        base_seed(int): The base seed for reproducibility.
        clf_n_jobs(int): Number of jobs to run in parallel for classification.
        reg_n_jobs(int): Number of jobs to run in parallel for regression.
    Returns:
        dict: A dictionary containing the metrics from the two-step hurdle model.
    """
    seed = _generate_seed(cutoff, inner, outer, repeat, base_seed)

    clf_model, clf_report = explorative_stage_one_classification(
        df, cutoff, inner, outer, seed=seed, clf_n_jobs=clf_n_jobs
    )
    reg_model, reg_report = explorative_stage_two_regression(
        df, cutoff, inner, outer, seed=seed, reg_n_jobs=reg_n_jobs
    )

    metrics_dict = {
        "combo_id": f"{cutoff}_{inner}_{outer}",
        "cutoff_quantile": float(cutoff),
        "n_inner_cv": int(inner),
        "n_outer_cv": int(outer),
        "repeat": float(repeat),
        "test_accuracy": float(clf_report.test_accuracy),
        "test_balanced_accuracy": float(clf_report.test_balanced_accuracy),
        "test_avg_precision": float(clf_report.test_avg_precision),
        "test_mcc": float(clf_report.test_mcc),
        "test_prevalence": float(clf_report.test_prevalence),
        "test_roc_auc": float(clf_report.test_roc_auc),
        "test_precision": float(clf_report.test_precision),
        "test_recall": float(clf_report.test_recall),
        "test_r2": float(reg_report.test_regression_r2),
        "test_mse": float(reg_report.test_regression_mse),
        "test_regression_mae": float(reg_report.test_regression_mae),
        "test_regression_rmse": float(reg_report.test_regression_rmse),
    }
    return metrics_dict, clf_report, reg_report


def _generate_seed(cutoff, n_inner_cv, n_outer_cv, repeat, base_seed=42):
    combo_id = f"{cutoff}_{n_inner_cv}_{n_outer_cv}"
    combo_hash = int(hashlib.sha256(combo_id.encode()).hexdigest(), 16) % 1_000_000
    return base_seed + combo_hash + int(repeat)
