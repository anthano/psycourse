import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def cca_prs_lipids_with_permtest(
    multimodal_lipid_subset_df: pd.DataFrame,
    prs_cols: list[str],
    lipid_cols: list[str],  # use your 16 lipid-class columns here
    n_components: int = 1,
    n_permutations: int = 10000,
    random_state: int = 42,
):
    """
    Perform Canonical Correlation Analysis (CCA) between PRS and lipid features
    with permutation testing.

    Args:
        multimodal_lipid_subset_df (pd.DataFrame): Lipid-subset multimodal Dataframe.
        prs_cols (list[str]): List of PRS feature columns.
        lipid_cols (list[str]): List of lipid feature columns.
        label_col (str): Name of the label column.
        n_components (int): Number of CCA components to compute.
        n_permutations (int): Number of permutations for significance testing.
        random_state (int): Random seed for reproducibility.
    """
    df = multimodal_lipid_subset_df.copy()
    df["is_cluster5"] = (df["true_label"] == 5).astype(int)

    label_col = "is_cluster5"
    use_cols = prs_cols + lipid_cols + [label_col]
    data = df[use_cols].dropna().copy()

    prs_scores, lipid_scores, can_corrs, cca = _cca(
        data, prs_cols, lipid_cols, n_components
    )

    binary_labels = data[label_col].to_numpy(dtype=int)

    cluster5_mask = binary_labels == 1
    non_cluster5_mask = binary_labels == 0

    prs_component1_scores_all = prs_scores[:, 0]
    prs_component1_scores_cluster5 = prs_component1_scores_all[cluster5_mask]
    prs_component1_scores_non_cluster5 = prs_component1_scores_all[non_cluster5_mask]

    t_statistic_observed = _welch_t_statistic(
        prs_component1_scores_cluster5,
        prs_component1_scores_non_cluster5,
    )

    # --- permutation test (two-sided) ---
    rng = np.random.default_rng(random_state)

    permuted_t_statistics = np.empty(n_permutations, dtype=float)

    for perm_idx in range(n_permutations):
        permuted_labels = rng.permutation(binary_labels)

        perm_cluster5_mask = permuted_labels == 1
        perm_non_cluster5_mask = permuted_labels == 0

        permuted_t_statistics[perm_idx] = _welch_t_statistic(
            prs_component1_scores_all[perm_cluster5_mask],
            prs_component1_scores_all[perm_non_cluster5_mask],
        )

    permutation_p_value_two_sided = (
        np.sum(np.abs(permuted_t_statistics) >= np.abs(t_statistic_observed)) + 1
    ) / (n_permutations + 1)

    return {
        "t_observed": float(t_statistic_observed),
        "p_perm_two_sided": float(permutation_p_value_two_sided),
        "canonical_correlations": can_corrs,
    }


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _cca(df, prs_cols, lipid_cols, n_components):
    prs_x1 = df[prs_cols].to_numpy(dtype=float)
    lipid_x2 = df[lipid_cols].to_numpy(dtype=float)

    scaled_prs_X1 = StandardScaler().fit_transform(prs_x1)
    scaled_lipid_X2 = StandardScaler().fit_transform(lipid_x2)

    n_components = int(
        min(n_components, scaled_prs_X1.shape[1], scaled_lipid_X2.shape[1])
    )
    cca = CCA(n_components=n_components, max_iter=5000)
    prs_scores, lipid_scores = cca.fit_transform(scaled_prs_X1, scaled_lipid_X2)
    # canonical scores (n x n_components)

    can_corrs = np.array(
        [
            np.corrcoef(prs_scores[:, k], lipid_scores[:, k])[0, 1]
            for k in range(n_components)
        ]
    )

    return prs_scores, lipid_scores, can_corrs, cca


def _welch_t_statistic(sample_a, sample_b):
    sample_a = np.asarray(sample_a)
    sample_b = np.asarray(sample_b)

    n_a = sample_a.size
    n_b = sample_b.size

    mean_a = sample_a.mean()
    mean_b = sample_b.mean()

    var_a = sample_a.var(ddof=1)
    var_b = sample_b.var(ddof=1)

    return (mean_a - mean_b) / np.sqrt(var_a / n_a + var_b / n_b)
