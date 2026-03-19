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
    prs_covariate_cols: list[str] | None = None,
    lipid_covariate_cols: list[str] | None = None,
    n_permutations: int = 20_000,
    random_state: int = 42,
) -> dict:
    """
    Incremental R² decomposition of PRS and lipid class contributions
    to severe psychosis subtype probability.

    Two parallel approaches are computed and returned:

    ── Residualized approach ────────────────────────────────────────────────────────

    Step 1 — Block-specific residualization:
      - PRS features: residualized against age, sex, and ancestry PCs (pc1–pc5)
      - Lipid features: residualized against sex, BMI, duration of illness, smoking
      - Outcome: residualized against the union of both covariate sets

    Step 2 — Five nested OLS models fitted on residualized features (intercept
    only, no additional covariates):
      M0: intercept only (shared baseline for M1, M2, M4)
      M1: residualized PRS
      M2: residualized lipids
      M3: residualized PRS + residualized lipids (conditional comparisons)
      M4: residualized PRS + residualized lipids (joint comparison vs M0;
          same fit as M3, distinct conceptual baseline)

    Step 3 — ΔR² values:
      dR2_prs          = R²(M1) − R²(M0)   [PRS alone]
      dR2_lip          = R²(M2) − R²(M0)   [lipids alone]
      dR2_prs_over_lip = R²(M3) − R²(M2)   [PRS beyond lipids]
      dR2_lip_over_prs = R²(M3) − R²(M1)   [lipids beyond PRS]
      dR2_joint        = R²(M4) − R²(M0)   [PRS + lipids jointly]

    ── Traditional validation approach (raw features, explicit covariates) ──────────

    Step 4 — Four traditional nested OLS models on raw (non-residualized) features,
    with the union covariate set entered explicitly in every model:
      M0_trad: outcome ~ union covariates
      M1_trad: outcome ~ union covariates + PRS
      M2_trad: outcome ~ union covariates + lipids
      M3_trad: outcome ~ union covariates + PRS + lipids

    Step 5 — Traditional ΔR² values:
      dR2_prs_trad          = R²(M1_trad) − R²(M0_trad)
      dR2_lip_trad          = R²(M2_trad) − R²(M0_trad)
      dR2_prs_over_lip_trad = R²(M3_trad) − R²(M2_trad)
      dR2_lip_over_prs_trad = R²(M3_trad) − R²(M1_trad)
      dR2_joint_trad        = R²(M3_trad) − R²(M0_trad)

    Both approaches share the same 20,000 outcome permutations (identical
    permutation index applied to y and y_raw in each replicate).

    Args:
        df: DataFrame containing all variables.
        prs_cols: PRS feature column names.
        lipid_cols: Lipid class feature column names.
        outcome_col: Continuous severity outcome column.
        prs_covariate_cols: Covariates to regress out of PRS features and the
            outcome. Defaults to age, sex, pc1–pc5.
        lipid_covariate_cols: Covariates to regress out of lipid features and
            the outcome. Defaults to sex, bmi, duration_illness, smoker.
        n_permutations: Permutations for p-value estimation.
        random_state: Random seed for reproducibility.

    Returns:
        dict with R² values, incremental R² values, and permutation p-values
        for both the residualized and traditional approaches.
    """
    if prs_covariate_cols is None:
        prs_covariate_cols = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]
    if lipid_covariate_cols is None:
        lipid_covariate_cols = ["sex", "bmi", "duration_illness", "smoker"]

    prs_covariate_cols = [c for c in prs_covariate_cols if c in df.columns]
    lipid_covariate_cols = [c for c in lipid_covariate_cols if c in df.columns]
    prs_cols = [c for c in prs_cols if c in df.columns]
    lipid_cols = [c for c in lipid_cols if c in df.columns]

    # Outcome is residualized against the union of both covariate sets
    outcome_covariate_cols = list(
        dict.fromkeys(prs_covariate_cols + lipid_covariate_cols)
    )

    all_cols = [outcome_col] + outcome_covariate_cols + prs_cols + lipid_cols
    df_model = df[all_cols].copy().dropna(subset=[outcome_col] + outcome_covariate_cols)

    # median-impute PRS and lipid missings, consistent with primary models
    for col in prs_cols + lipid_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    # one-hot encode categorical covariates (sex, smoker), drop_first to avoid
    # collinearity; add intercept column
    X_prs_cov = add_constant(
        pd.get_dummies(df_model[prs_covariate_cols], drop_first=True)
        .astype(float)
        .to_numpy()
    )
    X_lip_cov = add_constant(
        pd.get_dummies(df_model[lipid_covariate_cols], drop_first=True)
        .astype(float)
        .to_numpy()
    )
    X_outcome_cov = add_constant(
        pd.get_dummies(df_model[outcome_covariate_cols], drop_first=True)
        .astype(float)
        .to_numpy()
    )

    y_raw = df_model[outcome_col].to_numpy(float)
    X_prs_raw = df_model[prs_cols].to_numpy(float)
    X_lip_raw = df_model[lipid_cols].to_numpy(float)

    # ── Residualized approach ─────────────────────────────────────────────────────

    y = _residualize(y_raw, X_outcome_cov)
    X_prs = _residualize(X_prs_raw, X_prs_cov)
    X_lip = _residualize(X_lip_raw, X_lip_cov)

    # Pre-compute design matrices (intercept + residualized features)
    # M0: intercept only; M1: PRS; M2: lipids; M3/M4: PRS + lipids
    X_m0 = np.ones((len(y), 1))
    X_prs_const = add_constant(X_prs)
    X_lip_const = add_constant(X_lip)
    X_both_const = add_constant(np.column_stack([X_prs, X_lip]))

    r2_m0 = _ols_r2(y, X_m0)  # ≈ 0.0 for residualized y; explicit for clarity
    r2_prs = _ols_r2(y, X_prs_const)
    r2_lip = _ols_r2(y, X_lip_const)
    r2_both = _ols_r2(y, X_both_const)

    dr2_prs = r2_prs - r2_m0
    dr2_lip = r2_lip - r2_m0
    dr2_prs_over_lip = r2_both - r2_lip
    dr2_lip_over_prs = r2_both - r2_prs
    dr2_joint = r2_both - r2_m0

    # ── Traditional approach ──────────────────────────────────────────────────────

    # Design matrices: union covariates (already with intercept) + raw features
    X_trad_m1 = np.column_stack([X_outcome_cov, X_prs_raw])
    X_trad_m2 = np.column_stack([X_outcome_cov, X_lip_raw])
    X_trad_m3 = np.column_stack([X_outcome_cov, X_prs_raw, X_lip_raw])

    r2_trad_m0 = _ols_r2(y_raw, X_outcome_cov)
    r2_trad_m1 = _ols_r2(y_raw, X_trad_m1)
    r2_trad_m2 = _ols_r2(y_raw, X_trad_m2)
    r2_trad_m3 = _ols_r2(y_raw, X_trad_m3)

    dr2_prs_trad = r2_trad_m1 - r2_trad_m0
    dr2_lip_trad = r2_trad_m2 - r2_trad_m0
    dr2_prs_over_lip_trad = r2_trad_m3 - r2_trad_m2
    dr2_lip_over_prs_trad = r2_trad_m3 - r2_trad_m1
    dr2_joint_trad = r2_trad_m3 - r2_trad_m0

    # ── Permutation testing (shared permutation indices) ─────────────────────────

    rng = np.random.default_rng(random_state)
    null_dr2_prs = np.zeros(n_permutations)
    null_dr2_lip = np.zeros(n_permutations)
    null_dr2_prs_over_lip = np.zeros(n_permutations)
    null_dr2_lip_over_prs = np.zeros(n_permutations)
    null_dr2_joint = np.zeros(n_permutations)
    null_dr2_prs_trad = np.zeros(n_permutations)
    null_dr2_lip_trad = np.zeros(n_permutations)
    null_dr2_prs_over_lip_trad = np.zeros(n_permutations)
    null_dr2_lip_over_prs_trad = np.zeros(n_permutations)
    null_dr2_joint_trad = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Single permutation index applied to both y and y_raw
        perm_idx = rng.permutation(len(y))
        y_perm = y[perm_idx]
        y_raw_perm = y_raw[perm_idx]

        # Residualized nulls
        r2_m0_p = _ols_r2(y_perm, X_m0)
        r2_prs_p = _ols_r2(y_perm, X_prs_const)
        r2_lip_p = _ols_r2(y_perm, X_lip_const)
        r2_both_p = _ols_r2(y_perm, X_both_const)
        null_dr2_prs[i] = r2_prs_p - r2_m0_p
        null_dr2_lip[i] = r2_lip_p - r2_m0_p
        null_dr2_prs_over_lip[i] = r2_both_p - r2_lip_p
        null_dr2_lip_over_prs[i] = r2_both_p - r2_prs_p
        null_dr2_joint[i] = r2_both_p - r2_m0_p

        # Traditional nulls
        r2_trad_m0_p = _ols_r2(y_raw_perm, X_outcome_cov)
        r2_trad_m1_p = _ols_r2(y_raw_perm, X_trad_m1)
        r2_trad_m2_p = _ols_r2(y_raw_perm, X_trad_m2)
        r2_trad_m3_p = _ols_r2(y_raw_perm, X_trad_m3)
        null_dr2_prs_trad[i] = r2_trad_m1_p - r2_trad_m0_p
        null_dr2_lip_trad[i] = r2_trad_m2_p - r2_trad_m0_p
        null_dr2_prs_over_lip_trad[i] = r2_trad_m3_p - r2_trad_m2_p
        null_dr2_lip_over_prs_trad[i] = r2_trad_m3_p - r2_trad_m1_p
        null_dr2_joint_trad[i] = r2_trad_m3_p - r2_trad_m0_p

    return {
        # ── Residualized R² values ────────────────────────────────────────────────
        "R2_M0_resid": round(r2_m0, 6),
        "R2_prs_resid": round(r2_prs, 6),
        "R2_lip_resid": round(r2_lip, 6),
        "R2_both_resid": round(r2_both, 6),
        # ── Residualized ΔR² and p-values ─────────────────────────────────────────
        "dR2_prs": round(dr2_prs, 6),
        "dR2_lip": round(dr2_lip, 6),
        "dR2_prs_over_lip": round(dr2_prs_over_lip, 6),
        "dR2_lip_over_prs": round(dr2_lip_over_prs, 6),
        "dR2_joint": round(dr2_joint, 6),
        "p_perm_dR2_prs": round(_perm_pvalue(dr2_prs, null_dr2_prs), 4),
        "p_perm_dR2_lip": round(_perm_pvalue(dr2_lip, null_dr2_lip), 4),
        "p_perm_dR2_prs_over_lip": round(
            _perm_pvalue(dr2_prs_over_lip, null_dr2_prs_over_lip), 4
        ),
        "p_perm_dR2_lip_over_prs": round(
            _perm_pvalue(dr2_lip_over_prs, null_dr2_lip_over_prs), 4
        ),
        "p_perm_dR2_joint": round(_perm_pvalue(dr2_joint, null_dr2_joint), 4),
        # ── Traditional R² values ─────────────────────────────────────────────────
        "R2_M0_trad": round(r2_trad_m0, 6),
        "R2_M1_trad": round(r2_trad_m1, 6),
        "R2_M2_trad": round(r2_trad_m2, 6),
        "R2_M3_trad": round(r2_trad_m3, 6),
        # ── Traditional ΔR² and p-values ──────────────────────────────────────────
        "dR2_prs_trad": round(dr2_prs_trad, 6),
        "dR2_lip_trad": round(dr2_lip_trad, 6),
        "dR2_prs_over_lip_trad": round(dr2_prs_over_lip_trad, 6),
        "dR2_lip_over_prs_trad": round(dr2_lip_over_prs_trad, 6),
        "dR2_joint_trad": round(dr2_joint_trad, 6),
        "p_perm_dR2_prs_trad": round(_perm_pvalue(dr2_prs_trad, null_dr2_prs_trad), 4),
        "p_perm_dR2_lip_trad": round(_perm_pvalue(dr2_lip_trad, null_dr2_lip_trad), 4),
        "p_perm_dR2_prs_over_lip_trad": round(
            _perm_pvalue(dr2_prs_over_lip_trad, null_dr2_prs_over_lip_trad), 4
        ),
        "p_perm_dR2_lip_over_prs_trad": round(
            _perm_pvalue(dr2_lip_over_prs_trad, null_dr2_lip_over_prs_trad), 4
        ),
        "p_perm_dR2_joint_trad": round(
            _perm_pvalue(dr2_joint_trad, null_dr2_joint_trad), 4
        ),
        # ── Metadata ─────────────────────────────────────────────────────────────
        "n": len(y),
        "n_prs_features": len(prs_cols),
        "n_lipid_features": len(lipid_cols),
        "n_permutations": n_permutations,
    }


def make_r2_table(results: dict) -> pd.DataFrame:
    """
    Returns a publication-ready DataFrame comparing the residualized and
    traditional incremental R² results side by side.

    Columns are a two-level MultiIndex: the outer level is the approach
    ("Residualized" or "Traditional (raw)") and the inner level is the
    statistic ("ΔR²" or "p (permutation)").

    Args:
        results: dict returned by incremental_r2_decomposition.

    Returns:
        pd.DataFrame with ΔR² and permutation p-values for both approaches,
        indexed by comparison label.
    """
    comparisons = [
        "PRS alone",
        "Lipids alone",
        "PRS | Lipids",
        "Lipids | PRS",
        "PRS + Lipids jointly",
    ]

    resid_dr2 = [
        results["dR2_prs"],
        results["dR2_lip"],
        results["dR2_prs_over_lip"],
        results["dR2_lip_over_prs"],
        results["dR2_joint"],
    ]
    resid_p = [
        results["p_perm_dR2_prs"],
        results["p_perm_dR2_lip"],
        results["p_perm_dR2_prs_over_lip"],
        results["p_perm_dR2_lip_over_prs"],
        results["p_perm_dR2_joint"],
    ]
    trad_dr2 = [
        results["dR2_prs_trad"],
        results["dR2_lip_trad"],
        results["dR2_prs_over_lip_trad"],
        results["dR2_lip_over_prs_trad"],
        results["dR2_joint_trad"],
    ]
    trad_p = [
        results["p_perm_dR2_prs_trad"],
        results["p_perm_dR2_lip_trad"],
        results["p_perm_dR2_prs_over_lip_trad"],
        results["p_perm_dR2_lip_over_prs_trad"],
        results["p_perm_dR2_joint_trad"],
    ]

    col_idx = pd.MultiIndex.from_tuples(
        [
            ("Residualized", "ΔR²"),
            ("Residualized", "p (permutation)"),
            ("Traditional (raw)", "ΔR²"),
            ("Traditional (raw)", "p (permutation)"),
        ]
    )

    data = list(zip(resid_dr2, resid_p, trad_dr2, trad_p, strict=False))
    return pd.DataFrame(data, index=comparisons, columns=col_idx)


########################################################################################
# Individual Predictor ΔR² Table
########################################################################################


def individual_predictor_r2_table(
    df: pd.DataFrame,
    prs_predictors: list[str],
    lipid_predictors: list[str],
    outcome_col: str = "prob_class_5",
    prs_covariate_cols: list[str] | None = None,
    lipid_covariate_cols: list[str] | None = None,
    n_permutations: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute individual ΔR² and permutation p-values for each predictor using the
    residualized approach, consistent with ``incremental_r2_decomposition``.

    For each predictor the procedure is:
      1. Restrict the sample to rows with non-missing outcome and covariates
         (median-impute any remaining predictor missings, as in the main analysis).
      2. Residualize both the outcome and the predictor against the block-specific
         covariate set (PRS covariates for PRS predictors; lipid covariates for
         lipid predictors).
      3. Fit M0 (intercept only) and M1 (intercept + residualized predictor) on the
         residualized outcome.
      4. ΔR² = R²(M1) − R²(M0).
      5. Permutation p-value: permute the residualized outcome 20 000 times and
         record the proportion of null ΔR² values ≥ observed ΔR².

    Args:
        df: DataFrame containing all variables (use the lipid-subset frame so
            sample composition matches the main incremental R² analysis).
        prs_predictors: Individual PRS column names to test (e.g. SCZ_PRS, BD_PRS,
            Education_PRS).
        lipid_predictors: Individual lipid class column names to test (the enriched
            lipid class scores).
        outcome_col: Continuous severity outcome column.
        prs_covariate_cols: Covariates for PRS residualization.
            Defaults to age, sex, pc1–pc5.
        lipid_covariate_cols: Covariates for lipid residualization.
            Defaults to sex, bmi, duration_illness, smoker.
        n_permutations: Number of permutations for p-value estimation.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns: predictor, type, dR2, p_permutation.
    """
    if prs_covariate_cols is None:
        prs_covariate_cols = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]
    if lipid_covariate_cols is None:
        lipid_covariate_cols = ["sex", "bmi", "duration_illness", "smoker"]

    rows = []

    for pred in prs_predictors:
        row = _single_predictor_r2(
            df=df,
            predictor_col=pred,
            predictor_type="PRS",
            outcome_col=outcome_col,
            covariate_cols=prs_covariate_cols,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        rows.append(row)

    for pred in lipid_predictors:
        row = _single_predictor_r2(
            df=df,
            predictor_col=pred,
            predictor_type="Lipid",
            outcome_col=outcome_col,
            covariate_cols=lipid_covariate_cols,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=["predictor", "type", "dR2", "p_permutation"])


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _single_predictor_r2(
    df: pd.DataFrame,
    predictor_col: str,
    predictor_type: str,
    outcome_col: str,
    covariate_cols: list[str],
    n_permutations: int,
    random_state: int,
) -> tuple:
    """
    Compute ΔR² and permutation p-value for a single residualized predictor.

    Returns a tuple (predictor_col, predictor_type, dR2, p_permutation).
    """
    cov_cols = [c for c in covariate_cols if c in df.columns]

    if predictor_col not in df.columns:
        return (predictor_col, predictor_type, np.nan, np.nan)

    all_cols = [outcome_col] + cov_cols + [predictor_col]
    df_model = df[all_cols].copy().dropna(subset=[outcome_col] + cov_cols)

    # Median-impute predictor missings (consistent with main analysis)
    df_model[predictor_col] = df_model[predictor_col].fillna(
        df_model[predictor_col].median()
    )

    X_cov = add_constant(
        pd.get_dummies(df_model[cov_cols], drop_first=True).astype(float).to_numpy()
    )

    y_raw = df_model[outcome_col].to_numpy(float)
    x_raw = df_model[predictor_col].to_numpy(float).reshape(-1, 1)

    # Residualize outcome and predictor against the same covariate set
    y = _residualize(y_raw, X_cov)
    x = _residualize(x_raw, X_cov)

    X_m0 = np.ones((len(y), 1))
    X_m1 = add_constant(x)

    r2_m0 = _ols_r2(y, X_m0)
    r2_m1 = _ols_r2(y, X_m1)
    dr2 = r2_m1 - r2_m0

    # Permutation test — permute the residualized outcome
    rng = np.random.default_rng(random_state)
    null_dr2 = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = y[rng.permutation(len(y))]
        null_dr2[i] = _ols_r2(y_perm, X_m1) - _ols_r2(y_perm, X_m0)

    p_perm = _perm_pvalue(dr2, null_dr2)

    return (predictor_col, predictor_type, round(dr2, 6), round(p_perm, 4))


def _residualize(X_target: np.ndarray, X_cov: np.ndarray) -> np.ndarray:
    """
    Project X_target onto the null space of X_cov (which must include an
    intercept column). Works for a 1-D outcome vector or a 2-D feature matrix.
    Returns residuals with the same shape as X_target.
    """
    coefs, *_ = np.linalg.lstsq(X_cov, X_target, rcond=None)
    return X_target - X_cov @ coefs


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    return float(OLS(y, X).fit().rsquared)


def _perm_pvalue(observed: float, null_distribution: np.ndarray) -> float:
    """One-sided permutation p-value (proportion of nulls >= observed)."""
    return float((null_distribution >= observed).mean())
