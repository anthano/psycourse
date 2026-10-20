import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

########################################################################################
# Outcome definition sensitivity analysis
########################################################################################

# The primary outcome is the out-of-fold SVM probability of the severe psychosis
# subtype. As a sensitivity analysis, key analyses are re-run with the in-sample
# probabilities (model refitted on all labelled participants and applied to them).


def with_outcome(multimodal_df, probabilities):
    """
    Replace prob_class_5 with an alternative severe psychosis probability for each
    participant, so that the existing analysis functions can be re-run unchanged.

    Args:
        multimodal_df (pd.DataFrame): Analysis DataFrame indexed by participant id.
        probabilities (pd.DataFrame): Alternative SVM probabilities for all
            participants, indexed by participant id.
    Returns:
        df (pd.DataFrame): Copy of multimodal_df with the alternative outcome.
    """
    df = multimodal_df.copy()
    df["prob_class_5"] = probabilities["prob_class_5"].reindex(df.index)
    return df


def compare_outcomes(samples, probabilities, labels=("oof", "in_sample")):
    """
    Agreement between the primary and an alternative severe psychosis probability.

    Args:
        samples (dict[str, pd.DataFrame]): Analysis DataFrames (with the primary
            prob_class_5) by sample name, e.g. PRS sample and lipid subset.
        probabilities (pd.DataFrame): Alternative SVM probabilities.
        labels (tuple[str, str]): Names of the primary and alternative outcome.
    Returns:
        pd.DataFrame: One row per sample with n, Pearson r, Spearman rho, mean and
        SD of both scores, and the mean absolute difference.
    """
    primary_label, alternative_label = labels
    rows = []
    for name, df in samples.items():
        primary = df["prob_class_5"]
        alternative = probabilities["prob_class_5"].reindex(df.index)
        keep = primary.notna() & alternative.notna()
        primary, alternative = primary[keep], alternative[keep]
        rows.append(
            {
                "sample": name,
                "n": int(keep.sum()),
                "pearson_r": pearsonr(primary, alternative)[0],
                "spearman_rho": spearmanr(primary, alternative)[0],
                f"mean_{primary_label}": primary.mean(),
                f"mean_{alternative_label}": alternative.mean(),
                f"sd_{primary_label}": primary.std(),
                f"sd_{alternative_label}": alternative.std(),
                "mean_abs_difference": (primary - alternative).abs().mean(),
            }
        )
    return pd.DataFrame(rows).set_index("sample")


def compare_results(
    primary_results,
    alternative_results,
    columns=("coef", "FDR"),
    labels=("oof", "in_sample"),
):
    """
    Side-by-side table of a results DataFrame under the primary and an alternative
    outcome, with significance (FDR < 0.05) under each.

    Args:
        primary_results (pd.DataFrame): Results with the primary outcome.
        alternative_results (pd.DataFrame): Same analysis, alternative outcome.
        columns (tuple[str]): Result columns to compare.
        labels (tuple[str, str]): Names of the primary and alternative outcome.
    Returns:
        pd.DataFrame: Columns suffixed with each label plus sig_<label> columns,
        sorted by the primary FDR.
    """
    primary_label, alternative_label = labels
    columns = list(columns)
    table = primary_results[columns].join(
        alternative_results[columns],
        lsuffix=f"_{primary_label}",
        rsuffix=f"_{alternative_label}",
        how="outer",
    )
    for label in labels:
        table[f"sig_{label}"] = table[f"FDR_{label}"] < 0.05
    return table.sort_values(f"FDR_{primary_label}")


def compare_r2(primary_results, alternative_results, labels=("oof", "in_sample")):
    """
    Side-by-side residualized ΔR² and permutation p-values from
    incremental_r2_decomposition under the primary and an alternative outcome.
    """
    keys = ["dR2_prs", "dR2_lip", "dR2_prs_over_lip", "dR2_lip_over_prs", "dR2_joint"]
    columns = {}
    for label, results in zip(
        labels, [primary_results, alternative_results], strict=False
    ):
        columns[f"dR2_{label}"] = [results[k] for k in keys]
        columns[f"p_{label}"] = [results[f"p_perm_{k}"] for k in keys]
    table = pd.DataFrame(columns, index=keys)
    # shared component = dR2_prs + dR2_lip - dR2_joint (no permutation p-value)
    shared = table.loc["dR2_prs"] + table.loc["dR2_lip"] - table.loc["dR2_joint"]
    table.loc["shared"] = [
        shared[col] if col.startswith("dR2") else np.nan for col in table.columns
    ]
    return table
