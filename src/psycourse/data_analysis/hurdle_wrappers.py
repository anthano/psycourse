import hashlib

from psycourse.data_analysis.two_step_hurdle import (
    stage_one_classification,
    stage_two_regression,
)


def _seed_for(cutoff, n_inner_cv, n_outer_cv, base_seed=42):
    combo_id = f"{cutoff}_{n_inner_cv}_{n_outer_cv}"
    combo_hash = int(hashlib.sha256(combo_id.encode()).hexdigest(), 16) % 1_000_000
    return base_seed + combo_hash


def run_single_combo_repeat(
    df, cutoff, inner, outer, repeat, base_seed=42, clf_n_jobs=1, reg_n_jobs=1
):
    seed = _seed_for(cutoff, inner, outer, base_seed)

    clf_model, clf_report = stage_one_classification(
        df, cutoff, inner, outer, seed=seed, clf_n_jobs=clf_n_jobs
    )
    reg_model, reg_report = stage_two_regression(
        df, cutoff, inner, outer, seed=seed, reg_n_jobs=reg_n_jobs
    )

    return {
        "combo_id": f"{cutoff}_{inner}_{outer}",
        "cutoff_quantile": float(cutoff),
        "n_inner_cv": int(inner),
        "n_outer_cv": int(outer),
        "repeat": float(repeat),
        "test_accuracy": float(clf_report.test_accuracy),
        "test_roc_auc": float(clf_report.test_roc_auc),
        "test_precision": float(clf_report.test_precision),
        "test_recall": float(clf_report.test_recall),
        "test_r2": float(reg_report.test_regression_r2),
        "test_mse": float(reg_report.test_regression_mse),
        "permutation_pvalue": float(reg_report.permutation_pvalue),
    }
