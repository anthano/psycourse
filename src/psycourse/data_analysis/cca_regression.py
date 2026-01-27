import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def cca_regression_analysis(
    df: pd.DataFrame,
    prs_cols: list[str],
    lipid_class_cols: list[str],
    y_col: str = "prob_class_5",
    categorical_covars: tuple[str, ...] = ("sex", "smoker"),
    n_components: int = 1,
    n_perm_coupling: int = 10000,
    n_perm_regression: int = 20000,
    random_state: int = 42,
) -> dict:
    """
    Perform canonical correlation analysis (CCA) on PRS and lipid class data,
    controlling for covariates.
    Use the first CCA component scores to analyse association with
    cluster 5 probability via OLS regression, with permutation testing for significance.
    """

    prs_covar_cols = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]
    lipid_covar_cols = ["age", "sex", "bmi", "duration_illness", "smoker"]

    prs_cols = [col for col in prs_cols if col in df.columns]
    lipid_class_cols = [col for col in lipid_class_cols if col in df.columns]
    prs_covar_cols = [col for col in prs_covar_cols if col in df.columns]
    lipid_covar_cols = [col for col in lipid_covar_cols if col in df.columns]

    needed = list(
        dict.fromkeys(
            [y_col] + prs_cols + lipid_class_cols + prs_covar_cols + lipid_covar_cols
        )
    )
    imputed_df = _impute_df(df[needed].copy())

    # residualize blocks
    prs_resid = _residualize_block(
        imputed_df[prs_cols].astype(float),
        imputed_df[prs_covar_cols],
        categorical_covars,
    )
    lip_resid = _residualize_block(
        imputed_df[lipid_class_cols].astype(float),
        imputed_df[lipid_covar_cols],
        categorical_covars,
    )

    # standardize residuals
    X = StandardScaler().fit_transform(prs_resid.to_numpy())
    Y = StandardScaler().fit_transform(lip_resid.to_numpy())

    n_components = int(min(n_components, X.shape[1], Y.shape[1]))
    cca = CCA(n_components=n_components, max_iter=5000)
    U, V = cca.fit_transform(X, Y)

    can_corr = float(np.corrcoef(U[:, 0], V[:, 0])[0, 1])

    # orient comp1 to positive correlation for interpretability
    if can_corr < 0:
        V[:, 0] *= -1
        cca.y_loadings_[:, 0] *= -1
        can_corr = -can_corr

    # coupling permutation p-value (permute Y rows, refit CCA)
    rng = np.random.default_rng(random_state)
    null = np.empty(n_perm_coupling, dtype=float)
    for i in range(n_perm_coupling):
        idx = rng.permutation(Y.shape[0])
        cca_p = CCA(n_components=1, max_iter=5000)
        Up, Vp = cca_p.fit_transform(X, Y[idx, :])
        null[i] = np.corrcoef(Up[:, 0], Vp[:, 0])[0, 1]
    p_coupling = float(
        (1 + np.sum(np.abs(null) >= abs(can_corr))) / (1 + n_perm_coupling)
    )

    # build score df for regression
    scores_df = pd.DataFrame(
        {
            "PRS_CCA_Component_1": U[:, 0],
            "Lipid_CCA_Component_1": V[:, 0],
            y_col: imputed_df[y_col].to_numpy(dtype=float),
        },
        index=imputed_df.index,
    )

    # regress continuous probability on each score
    res_prs, t_prs = _fit_ols_get_t(scores_df, y_col, "PRS_CCA_Component_1")
    res_lip, t_lip = _fit_ols_get_t(scores_df, y_col, "Lipid_CCA_Component_1")

    p_perm_prs = _permute_regression_pval(
        scores_df=scores_df,
        y_col=y_col,
        score_col="PRS_CCA_Component_1",
        t_observed=t_prs,
        n_permutations=n_perm_regression,
        random_state=random_state,
    )
    p_perm_lip = _permute_regression_pval(
        scores_df=scores_df,
        y_col=y_col,
        score_col="Lipid_CCA_Component_1",
        t_observed=t_lip,
        n_permutations=n_perm_regression,
        random_state=random_state,
    )

    prs_loadings = pd.Series(
        cca.x_loadings_[:, 0], index=prs_cols, name="cca_loading"
    ).sort_values(key=np.abs, ascending=False)
    lipid_loadings = pd.Series(
        cca.y_loadings_[:, 0], index=lipid_class_cols, name="cca_loading"
    ).sort_values(key=np.abs, ascending=False)

    results_dict = {
        "cca": cca,
        "canonical_corr_comp1": can_corr,
        "p_coupling_perm_comp1": p_coupling,
        "scores": scores_df,
        "reg_prob_on_prs_score": res_prs,
        "p_perm_prob_on_prs_score": float(p_perm_prs),
        "reg_prob_on_lipid_score": res_lip,
        "p_perm_prob_on_lipid_score": float(p_perm_lip),
        "prs_loadings": prs_loadings,
        "lipid_class_loadings": lipid_loadings,
    }

    return results_dict


def _fit_ols_get_t(scores_df: pd.DataFrame, y_col: str, score_col: str):
    res = smf.ols(f"{y_col} ~ {score_col}", data=scores_df).fit(cov_type="HC3")
    coef = res.params.get(score_col, np.nan)
    se = res.bse.get(score_col, np.nan)
    t_stat = coef / se if np.isfinite(coef) and np.isfinite(se) and se != 0 else np.nan
    return res, float(t_stat)


def _permute_regression_pval(
    scores_df: pd.DataFrame,
    y_col: str,
    score_col: str,
    t_observed: float,
    n_permutations: int = 20000,
    random_state: int = 0,
) -> float:
    rng = np.random.default_rng(random_state)
    y = scores_df[y_col].to_numpy()
    t_perm = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):
        perm_df = scores_df.copy()
        perm_df[y_col] = rng.permutation(y)
        _, t_perm[i] = _fit_ols_get_t(perm_df, y_col, score_col)

    t_perm = t_perm[np.isfinite(t_perm)]
    if not np.isfinite(t_observed) or t_perm.size == 0:
        return float("nan")

    return float((1 + np.sum(np.abs(t_perm) >= abs(t_observed))) / (1 + t_perm.size))


def _impute_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = df[col].median()
            if pd.isna(fill):
                fill = 0.0
        else:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else ""
        df[col] = df[col].fillna(fill)
    return df


def _residualize_block(
    block: pd.DataFrame,
    cov: pd.DataFrame,
    categorical_covars: tuple[str, ...],
) -> pd.DataFrame:
    cov = cov.copy()
    for col in categorical_covars:
        if col in cov.columns:
            cov[col] = cov[col].astype("category")

    C = pd.get_dummies(cov, drop_first=False)
    C.insert(0, "Intercept", 1.0)

    X = C.to_numpy(dtype=float)
    Y = block.to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    return pd.DataFrame(resid, index=block.index, columns=block.columns)
