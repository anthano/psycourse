import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def lipid_class_enrichment_gsea(
    results_df,
    annot_df,
    score_col="coef",  # signed statistic from univariate_lipid_regression
    class_col="class",  # column in annot_df with class labels
    n_perms=5000,
    random_state=42,
    min_in_class=5,
    weight=1.0,  # GSEA weight parameter (p)
):
    """
    GSEA-style enrichment test for lipid classes with running-sum enrichment scores(ES).

    Args:
        results_df (pd.DataFrame): output of univariate_lipid_regression (index = lipid)
                must contain column "coef" which is the score column here.
        annot_df (pd.DataFrame):  DataFrame with lipid -> class mapping
          (index = lipid, col = class_col).

    Returns:
        out (pd.DataFrame):
        DataFrame containing the result statistics of enrichment analysis.

    """

    rng = np.random.default_rng(random_state)

    # harmonise class column name and join
    annot_use = annot_df[[class_col]].rename(columns={class_col: "class"})
    merged = results_df[[score_col]].join(annot_use, how="inner").dropna()
    if merged.empty:
        return pd.DataFrame()

    # signed scores for ranking
    scores = merged[score_col].to_numpy(dtype=float)

    # rank high -> low
    order = np.argsort(-scores)
    merged = merged.iloc[order].copy()
    scores = scores[order]
    all_classes = merged["class"].to_numpy()

    # precompute class masks
    lipid_classes = np.unique(all_classes)
    class_masks = {}
    for lipid_class in lipid_classes:
        mask = all_classes == lipid_class
        if mask.sum() >= min_in_class:
            class_masks[lipid_class] = mask

    if not class_masks:
        return pd.DataFrame()

    # observed ES per class
    obs_es = {
        cl: _gsea_es(scores, mask, weight=weight) for cl, mask in class_masks.items()
    }

    # permutation nulls: shuffle scores, keep class membership fixed
    null_es = {cl: [] for cl in class_masks.keys()}

    for _ in range(n_perms):
        perm_scores = rng.permutation(scores)
        for lipid_class, mask in class_masks.items():
            enrichment_score = _gsea_es(perm_scores, mask, weight=weight)
            null_es[lipid_class].append(enrichment_score)

    # p-values + NES
    rows = []
    for lipid_class, enrichment_score in obs_es.items():
        es_null = np.asarray(null_es[lipid_class], dtype=float)

        if enrichment_score >= 0:
            pval = (np.sum(es_null >= enrichment_score) + 1.0) / (len(es_null) + 1.0)
            pos_null = es_null[es_null > 0]
            nes = enrichment_score / np.mean(pos_null) if len(pos_null) > 0 else np.nan
        else:
            pval = (np.sum(es_null <= enrichment_score) + 1.0) / (len(es_null) + 1.0)
            neg_null = es_null[es_null < 0]
            nes = (
                enrichment_score / np.abs(np.mean(neg_null))
                if len(neg_null) > 0
                else np.nan
            )

        rows.append(
            {
                "class": lipid_class,
                "n_in_class": class_masks[lipid_class].sum(),
                "ES": enrichment_score,
                "NES": nes,
                "pval": pval,
            }
        )

    out = pd.DataFrame(rows).set_index("class")
    if out.empty:
        return out

    out["FDR"] = multipletests(out["pval"], method="fdr_bh")[1]
    out["-log10(FDR)"] = -np.log10(np.clip(out["FDR"], np.finfo(float).tiny, None))
    return out.sort_values("FDR")


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _gsea_es(
    ranked_scores: np.ndarray,
    set_membership_mask: np.ndarray,
    weight: float = 1.0,
) -> float:
    """
    Compute the running-sum GSEA enrichment score (ES) for a single set.

    Args:
    ranked_scores (np.ndarray): 1D array of scores,
    already ordered from highest to lowest.

    set_membership_mask (np.ndarray):
    Boolean array of the same length as `ranked_scores`, where True indicates
        that the position belongs to the set of interest.

    weight (float, optional):
      GSEA weighting parameter (p). weight=0 -> unweighted, weight=1 -> standard.

    Returns:
    es (float): Enrichment score (ES) for the set.
    """
    scores = np.asarray(ranked_scores, dtype=float)
    in_set = np.asarray(set_membership_mask, dtype=bool)

    n_total = scores.size
    n_in_set = in_set.sum()
    if n_in_set in (0, n_total):
        return 0.0

    # weighted absolute scores for "hit" contributions
    abs_weighted_scores = np.abs(scores) ** weight
    hit_weight_sum = abs_weighted_scores[in_set].sum()

    if hit_weight_sum == 0:
        hit_step_contributions = np.zeros_like(scores)
        hit_step_contributions[in_set] = 1.0 / n_in_set
    else:
        hit_step_contributions = np.zeros_like(scores)
        hit_step_contributions[in_set] = abs_weighted_scores[in_set] / hit_weight_sum

    miss_step = 1.0 / (n_total - n_in_set)

    running_sum = 0.0
    max_running_sum = -np.inf
    min_running_sum = np.inf

    for idx in range(n_total):
        if in_set[idx]:
            running_sum += hit_step_contributions[idx]
        else:
            running_sum -= miss_step

        max_running_sum = max(running_sum, max_running_sum)
        min_running_sum = min(running_sum, min_running_sum)

    return (
        max_running_sum
        if abs(max_running_sum) >= abs(min_running_sum)
        else min_running_sum
    )
