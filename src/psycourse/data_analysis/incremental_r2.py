import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

########################################################################################
# Incremental R² Decomposition
########################################################################################


def incremental_r2_decomposition(
    df: pd.DataFrame,
    prs_cols: list[str],
    lipid_cols: list[str],
    outcome_col: str = "prob_class_5",
    covariate_cols: list[str] | None = None,
    n_permutations: int = 20_000,
    random_state: int = 42,
) -> dict:
    """
    Incremental R² decomposition of PRS and lipid class contributions
    to severe psychosis subtype probability.

    Fits four nested OLS models (M0: covariates, M1: cov+PRS, M2: cov+lipids,
    M3: cov+PRS+lipids) and computes ΔR² for each block with permutation p-values.

    Args:
        df: DataFrame containing all variables.
        prs_cols: PRS feature column names.
        lipid_cols: Lipid class feature column names.
        outcome_col: Continuous severity outcome column.
        covariate_cols: Baseline covariates. Defaults to age, sex, BMI,
            duration_illness, smoker, pc1-pc5.
        n_permutations: Permutations for p-value estimation.
        random_state: Random seed for reproducibility.

    Returns:
        dict with R² values, incremental R² values, and permutation p-values.
    """
    if covariate_cols is None:
        covariate_cols = [
            "age",
            "sex",
            "bmi",
            "duration_illness",
            "smoker",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
        ]

    covariate_cols = [c for c in covariate_cols if c in df.columns]
    prs_cols = [c for c in prs_cols if c in df.columns]
    lipid_cols = [c for c in lipid_cols if c in df.columns]

    all_cols = [outcome_col] + covariate_cols + prs_cols + lipid_cols
    df_model = df[all_cols].copy().dropna(subset=[outcome_col] + covariate_cols)

    # median-impute PRS and lipid missings, consistent with primary models
    for col in prs_cols + lipid_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    y = df_model[outcome_col].to_numpy(float)
    # one-hot encode categorical covariates (sex, smoker), drop_first to avoid
    # #collinearity
    cov_encoded = pd.get_dummies(df_model[covariate_cols], drop_first=True).astype(
        float
    )
    X_cov = add_constant(cov_encoded.to_numpy(float))
    X_prs = df_model[prs_cols].to_numpy(float)
    X_lip = df_model[lipid_cols].to_numpy(float)

    r2_m0 = _ols_r2(y, X_cov)
    r2_m1 = _ols_r2(y, np.column_stack([X_cov, X_prs]))
    r2_m2 = _ols_r2(y, np.column_stack([X_cov, X_lip]))
    r2_m3 = _ols_r2(y, np.column_stack([X_cov, X_prs, X_lip]))

    dr2_prs = r2_m1 - r2_m0
    dr2_lip = r2_m2 - r2_m0
    dr2_prs_over_lip = r2_m3 - r2_m2
    dr2_lip_over_prs = r2_m3 - r2_m1

    rng = np.random.default_rng(random_state)
    null_dr2_prs = np.zeros(n_permutations)
    null_dr2_lip = np.zeros(n_permutations)
    null_dr2_prs_over_lip = np.zeros(n_permutations)
    null_dr2_lip_over_prs = np.zeros(n_permutations)

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        r2_m0_p = _ols_r2(y_perm, X_cov)
        r2_m1_p = _ols_r2(y_perm, np.column_stack([X_cov, X_prs]))
        r2_m2_p = _ols_r2(y_perm, np.column_stack([X_cov, X_lip]))
        r2_m3_p = _ols_r2(y_perm, np.column_stack([X_cov, X_prs, X_lip]))
        null_dr2_prs[i] = r2_m1_p - r2_m0_p
        null_dr2_lip[i] = r2_m2_p - r2_m0_p
        null_dr2_prs_over_lip[i] = r2_m3_p - r2_m2_p
        null_dr2_lip_over_prs[i] = r2_m3_p - r2_m1_p

    return {
        "R2_M0_covariates": round(r2_m0, 6),
        "R2_M1_cov_prs": round(r2_m1, 6),
        "R2_M2_cov_lip": round(r2_m2, 6),
        "R2_M3_cov_prs_lip": round(r2_m3, 6),
        "dR2_prs": round(dr2_prs, 6),
        "dR2_lip": round(dr2_lip, 6),
        "dR2_prs_over_lip": round(dr2_prs_over_lip, 6),
        "dR2_lip_over_prs": round(dr2_lip_over_prs, 6),
        "p_perm_dR2_prs": round(_perm_pvalue(dr2_prs, null_dr2_prs), 4),
        "p_perm_dR2_lip": round(_perm_pvalue(dr2_lip, null_dr2_lip), 4),
        "p_perm_dR2_prs_over_lip": round(
            _perm_pvalue(dr2_prs_over_lip, null_dr2_prs_over_lip), 4
        ),
        "p_perm_dR2_lip_over_prs": round(
            _perm_pvalue(dr2_lip_over_prs, null_dr2_lip_over_prs), 4
        ),
        "n": len(y),
        "n_prs_features": len(prs_cols),
        "n_lipid_features": len(lipid_cols),
        "n_permutations": n_permutations,
    }


def make_r2_table(results: dict) -> pd.DataFrame:
    """
    Returns a publication-ready DataFrame summarising incremental R² results.

    Args:
        results: dict returned by incremental_r2_decomposition.

    Returns:
        pd.DataFrame with ΔR² and permutation p-values per comparison.
    """
    rows = [
        {
            "Comparison": "PRS beyond covariates",
            "ΔR²": results["dR2_prs"],
            "p (permutation)": results["p_perm_dR2_prs"],
        },
        {
            "Comparison": "Lipids beyond covariates",
            "ΔR²": results["dR2_lip"],
            "p (permutation)": results["p_perm_dR2_lip"],
        },
        {
            "Comparison": "PRS beyond covariates + lipids",
            "ΔR²": results["dR2_prs_over_lip"],
            "p (permutation)": results["p_perm_dR2_prs_over_lip"],
        },
        {
            "Comparison": "Lipids beyond covariates + PRS",
            "ΔR²": results["dR2_lip_over_prs"],
            "p (permutation)": results["p_perm_dR2_lip_over_prs"],
        },
    ]
    return pd.DataFrame(rows).set_index("Comparison")


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    return float(OLS(y, X).fit().rsquared)


def _perm_pvalue(observed: float, null_distribution: np.ndarray) -> float:
    """One-sided permutation p-value (proportion of nulls >= observed)."""
    return float((null_distribution >= observed).mean())
