from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def cca_prs_lipids_regression(
    multimodal_lipid_subset_df: pd.DataFrame,
    prs_cols: list[str],
    lipid_cols: list[str],
    n_components: int = 1,
    score_col: str = "PRS_CCA_Component_1",  # or "Lipid_CCA_Component_1"
    covar_cols: list[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
    y_col: str = "prob_class_5",
    n_permutations: int = 10000,
    random_state: int = 0,
):
    df = multimodal_lipid_subset_df.copy()

    use_cols = prs_cols + lipid_cols
    data = df[use_cols].dropna().copy()

    cca_result_df, cca = _perform_cca(data, prs_cols, lipid_cols, n_components)
    obs_corrs, p_corrs = _permute_cca_corr_pvals(
        df=data,
        prs_cols=prs_cols,
        lipid_cols=lipid_cols,
        n_components=n_components,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    for k in range(len(p_corrs)):
        cca_result_df[f"CCA_Correlation_p_perm_{k+1}"] = p_corrs[k]

    combined_df = _join_cca_results_with_df(multimodal_lipid_subset_df, cca_result_df)

    prs_loadings_df, lipid_loadings_df = _get_loadings(cca, prs_cols, lipid_cols)

    if categorical_cols is None:
        categorical_cols = ("sex", "smoker")
    if covar_cols is None:
        if score_col.startswith("PRS_CCA_Component"):
            covar_cols = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]
        elif score_col.startswith("Lipid_CCA_Component"):
            covar_cols = ["age", "sex", "bmi", "duration_illness", "smoker"]

    regression_result_df, t_obs = _perform_regression(
        combined_df=combined_df,
        y_col=y_col,
        score_col=score_col,
        covar_cols=covar_cols,
        categorical_cols=categorical_cols,
    )

    p_perm = _permute_label_pval(
        combined_df=combined_df,
        t_observed=t_obs,
        y_col=y_col,
        score_col=score_col,
        covar_cols=covar_cols,
        categorical_cols=categorical_cols,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    regression_result_df["p_perm_two_sided"] = p_perm
    return cca, prs_loadings_df, lipid_loadings_df, regression_result_df


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _perform_cca(df, prs_cols, lipid_cols, n_components):
    prs_x1 = df[prs_cols].to_numpy(dtype=float)
    lipid_x2 = df[lipid_cols].to_numpy(dtype=float)

    scaled_prs_X1 = StandardScaler().fit_transform(prs_x1)
    scaled_lipid_X2 = StandardScaler().fit_transform(lipid_x2)

    n_components = int(
        min(n_components, scaled_prs_X1.shape[1], scaled_lipid_X2.shape[1])
    )
    cca = CCA(n_components=n_components, max_iter=5000)
    prs_scores, lipid_scores = cca.fit_transform(scaled_prs_X1, scaled_lipid_X2)

    cols = {}
    for k in range(n_components):
        cols[f"PRS_CCA_Component_{k+1}"] = prs_scores[:, k]
        cols[f"Lipid_CCA_Component_{k+1}"] = lipid_scores[:, k]
        cols[f"CCA_Correlation_{k+1}"] = np.corrcoef(
            prs_scores[:, k], lipid_scores[:, k]
        )[0, 1]

    result_df = pd.DataFrame(cols, index=df.index)
    return result_df, cca


def _join_cca_results_with_df(multimodal_df, result_df):
    return multimodal_df.join(result_df, how="inner")


def _build_formula(y_col, score_col, covar_cols, categorical_cols):
    terms = [score_col]
    for col in covar_cols:
        if col in categorical_cols:
            terms.append(f"C({col})")
        else:
            terms.append(col)
    return f"{y_col} ~ " + " + ".join(terms)


def _fit_glm_get_t(subset, y_col, score_col, covar_cols, categorical_cols):
    formula = _build_formula(y_col, score_col, covar_cols, categorical_cols)
    res = smf.glm(formula=formula, data=subset).fit(cov_type="HC3")
    coef = res.params.get(score_col, np.nan)
    se = res.bse.get(score_col, np.nan)
    t_stat = coef / se if np.isfinite(coef) and np.isfinite(se) and se != 0 else np.nan
    return res, t_stat


def _perform_regression(combined_df, y_col, score_col, covar_cols, categorical_cols):
    cols = [score_col, y_col, *covar_cols]
    subset = combined_df[cols].dropna()

    res, t_obs = _fit_glm_get_t(subset, y_col, score_col, covar_cols, categorical_cols)

    coef = res.params.get(score_col, np.nan)
    se = res.bse.get(score_col, np.nan)
    pval = res.pvalues.get(score_col, np.nan)
    ci_lower, ci_upper = res.conf_int().loc[score_col].tolist()

    out = pd.DataFrame(
        [
            {
                "prs": score_col,
                "coef": coef,
                "se": se,
                "ci_low": ci_lower,
                "ci_high": ci_upper,
                "pval": pval,
            }
        ]
    ).set_index("prs")

    out["FDR"] = multipletests(out["pval"], method="fdr_bh")[1]
    out["log10_FDR"] = -np.log10(np.clip(out["FDR"], np.finfo(float).tiny, None))
    out = out.sort_values(by="FDR")
    return out, t_obs


def _permute_label_pval(
    combined_df,
    t_observed,
    y_col,
    score_col,
    covar_cols,
    categorical_cols,
    n_permutations=10000,
    random_state=0,
):
    cols = [score_col, y_col, *covar_cols]
    subset = combined_df[cols].dropna().copy()

    rng = np.random.default_rng(random_state)
    y = subset[y_col].to_numpy()

    t_perm = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        subset_perm = subset.copy()
        subset_perm[y_col] = rng.permutation(y)
        _, t_perm[i] = _fit_glm_get_t(
            subset_perm, y_col, score_col, covar_cols, categorical_cols
        )

    t_perm = t_perm[np.isfinite(t_perm)]
    if not np.isfinite(t_observed) or t_perm.size == 0:
        return np.nan

    return float((1 + np.sum(np.abs(t_perm) >= abs(t_observed))) / (1 + t_perm.size))


def _permute_cca_corr_pvals(
    df,
    prs_cols,
    lipid_cols,
    n_components,
    n_permutations=5000,
    random_state=0,
):
    X1 = StandardScaler().fit_transform(df[prs_cols].to_numpy(dtype=float))
    X2 = StandardScaler().fit_transform(df[lipid_cols].to_numpy(dtype=float))

    n_components = int(min(n_components, X1.shape[1], X2.shape[1]))
    rng = np.random.default_rng(random_state)

    cca = CCA(n_components=n_components, max_iter=5000)
    U, V = cca.fit_transform(X1, X2)
    obs = np.array(
        [np.corrcoef(U[:, k], V[:, k])[0, 1] for k in range(n_components)], dtype=float
    )

    null = np.empty((n_permutations, n_components), dtype=float)
    for i in range(n_permutations):
        idx = rng.permutation(X2.shape[0])
        cca_p = CCA(n_components=n_components, max_iter=5000)
        Up, Vp = cca_p.fit_transform(X1, X2[idx, :])
        null[i, :] = [
            np.corrcoef(Up[:, k], Vp[:, k])[0, 1] for k in range(n_components)
        ]

    pvals = np.empty(n_components, dtype=float)
    abs_null = np.abs(null)
    abs_obs = np.abs(obs)
    for k in range(n_components):
        pvals[k] = (1 + np.sum(abs_null[:, k] >= abs_obs[k])) / (1 + n_permutations)

    return obs, pvals


def _get_loadings(cca, prs_cols, lipid_cols, component=1):
    lipid_cols = [col.removeprefix("class_") for col in lipid_cols]

    k = component - 1
    prs_vec = cca.x_loadings_[:, k]
    lipid_vec = cca.y_loadings_[:, k]

    prs_loadings_df = pd.DataFrame(
        {"cca_loading": prs_vec}, index=pd.Index(prs_cols, name="prs")
    )
    lipid_loadings_df = pd.DataFrame(
        {"cca_loading": lipid_vec}, index=pd.Index(lipid_cols, name="class")
    )
    return prs_loadings_df, lipid_loadings_df
