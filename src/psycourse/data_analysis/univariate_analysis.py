import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

############## PRS ################


def univariate_prs_regression(multimodal_df):
    """
    Univariate GLMs of prob_class_5 on each PRS, adjusted for covariates.
    Returns coef, SE, 95% CI, p, FDR, and -log10(FDR).
    Uses HC3 robust covariance (with a safe fallback for older statsmodels).
    """
    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]
    rows = []

    for prs in prs_columns:
        cols = [
            prs,
            "prob_class_5",
            "age",
            "sex",
            "bmi",
            "diagnosis",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
            "pc6",
            "pc7",
            "pc8",
            "pc9",
            "pc10",
        ]
        subset = multimodal_df[cols].dropna()

        formula = (
            f"prob_class_5 ~ {prs} + age + C(sex) + bmi  + "
            "pc1 + pc2 + pc3 + pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10"
        )

        model = smf.glm(formula=formula, data=subset)

        result = model.fit(cov_type="HC3")  # correct for heteroscedasticity

        coef = result.params.get(prs, np.nan)
        se = result.bse.get(prs, np.nan)
        pval = result.pvalues.get(prs, np.nan)

        # 95% CI (respects robust covariance)
        ci_lower, ci_upper = result.conf_int().loc[prs].tolist()

        rows.append(
            {
                "prs": prs,
                "coef": coef,
                "se": se,
                "ci_low": ci_lower,
                "ci_high": ci_upper,
                "pval": pval,
            }
        )

    df = pd.DataFrame(rows).set_index("prs")
    df["FDR"] = multipletests(df["pval"], method="fdr_bh")[1]
    df["log10_FDR"] = -np.log10(np.clip(df["FDR"], np.finfo(float).tiny, None))
    df = df.sort_values(by="FDR")
    return df


def univariate_prs_regression_cov_diag(multimodal_df):
    """
    Perform univariate regression analysis for PRS (Polygenic Risk Scores) against
    the probability of class 5 WITH diagnosis as an added covariate.
    This function iterates through all PRS columns in the provided multimodal DataFrame,
    fits a GLM model for each, and returns a DataFrame with coefficients, p-values,
    and FDR-corrected p-values.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
    Returns:
        pd.DataFrame: DataFrame with PRS names as index, coefficients, p-values,
        and FDR-corrected p-values.
    """

    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]

    prs_records = []
    for prs in prs_columns:
        subset = multimodal_df[
            [prs, "prob_class_5", "age", "sex", "bmi", "diagnosis"]
        ].dropna()

        formula = f"prob_class_5 ~ {prs} + age + C(sex) + bmi + C(diagnosis)"
        model = smf.glm(formula, data=subset).fit()
        coef = model.params.get(prs, np.nan)
        pval = model.pvalues.get(prs, np.nan)
        prs_records.append({"prs": prs, "coef": coef, "pval": pval})

    results_df_cov_diag = pd.DataFrame(prs_records).set_index("prs")
    results_df_cov_diag["FDR"] = multipletests(
        results_df_cov_diag["pval"], method="fdr_bh"
    )[1]
    results_df_cov_diag["log10_FDR"] = -np.log10(results_df_cov_diag["FDR"])

    return results_df_cov_diag


def prs_cv_delta_mse(multimodal_df, n_splits=5, random_state=42):
    """
    For each PRS, compute the % reduction in MSE on held-out folds when adding the PRS
    to a covariates-only linear model predicting prob_class_5.

    Returns a DataFrame with mean and SD across folds.
    """
    prs_cols = [c for c in multimodal_df.columns if c.endswith("PRS")]
    covars = [
        "age",
        "sex",
        "bmi",
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pc5",
        "pc6",
        "pc7",
        "pc8",
        "pc9",
        "pc10",
    ]
    outcome = "prob_class_5"

    # keep only needed cols and drop rows with any NA
    cols_needed = [outcome] + covars + prs_cols
    df = multimodal_df[cols_needed].dropna().copy()

    # one-hot encode sex (drop_first to avoid collinearity)
    df_enc = pd.get_dummies(df, columns=["sex"], drop_first=True)

    y = df_enc[outcome].to_numpy()
    covar_cols_enc = [c for c in df_enc.columns if c not in prs_cols + [outcome]]
    Xc = df_enc[covar_cols_enc].to_numpy()

    # CV splitter
    k = min(max(2, n_splits), len(df_enc))  # at least 2 folds, at most n samples
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)

    results = []
    for prs in prs_cols:
        Xi = np.column_stack([Xc, df_enc[[prs]].to_numpy()])

        mse_base_folds, mse_full_folds, deltas_pct = [], [], []
        for tr, te in cv.split(Xc):
            # baseline
            base = LinearRegression().fit(Xc[tr], y[tr])
            yb = base.predict(Xc[te])
            # full
            full = LinearRegression().fit(Xi[tr], y[tr])
            yf = full.predict(Xi[te])

            # (optional) clip predictions to [0,1] since y is a probability
            yb = np.clip(yb, 0, 1)
            yf = np.clip(yf, 0, 1)

            mse_b = mean_squared_error(y[te], yb)
            mse_f = mean_squared_error(y[te], yf)

            mse_base_folds.append(mse_b)
            mse_full_folds.append(mse_f)

            # % reduction; negative means worse with PRS
            deltas_pct.append(100.0 * (mse_b - mse_f) / mse_b if mse_b > 0 else 0.0)

        results.append(
            {
                "PRS": prs,
                "delta_mse_pct_mean": float(np.mean(deltas_pct)),
                "delta_mse_pct_std": float(np.std(deltas_pct, ddof=1))
                if k > 1
                else np.nan,
                "mse_base_mean": float(np.mean(mse_base_folds)),
                "mse_full_mean": float(np.mean(mse_full_folds)),
                "n": len(df_enc),
                "kfolds": k,
            }
        )

    out = (
        pd.DataFrame(results)
        .set_index("PRS")
        .sort_values("delta_mse_pct_mean", ascending=False)
    )
    return out


def univariate_prs_ancova(multimodal_df):
    """
    Perform univariate ANCOVA analysis for PRS (Polygenic Risk Scores) against
    the probability of class 5, focusing on the top and bottom 50 extreme values.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
    Returns:
        pd.DataFrame: DataFrame with PRS names as index, coefficients, p-values,
        and FDR-corrected p-values.
    """

    clean_df = multimodal_df.dropna()
    thresholds = [50, 100, 120]
    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]

    results_dfs = {}

    for threshold in thresholds:
        top_idx = clean_df["prob_class_5"].nlargest(threshold).index
        bottom_idx = clean_df["prob_class_5"].nsmallest(threshold).index
        extreme_df = clean_df.loc[top_idx.union(bottom_idx)].copy()
        extreme_df["extreme_group"] = 0  # default: bottom
        extreme_df.loc[top_idx, "extreme_group"] = 1  # mark top
        prs_records = []
        subset_dfs = []

        for prs in prs_columns:
            subset = extreme_df[
                [prs, "prob_class_5", "extreme_group", "age", "sex", "bmi", "diagnosis"]
            ].dropna()
            formula = f"{prs} ~ C(extreme_group) + age + C(sex) + bmi"
            model = smf.ols(formula, data=subset).fit()
            group_coef = model.params.get("C(extreme_group)[T.1]", np.nan)
            group_pval = model.pvalues.get("C(extreme_group)[T.1]", np.nan)

            prs_records.append({"prs": prs, "coef": group_coef, "pval": group_pval})
            subset_dfs.append(subset)

        results_df = pd.DataFrame(prs_records).set_index("prs")
        results_df["FDR"] = multipletests(results_df["pval"], method="fdr_bh")[1]
        results_df["log10_FDR"] = -np.log10(results_df["FDR"])
        results_df = results_df.sort_values(by="FDR")
        results_dfs[threshold] = results_df

    return results_dfs


######################### Lipids #########################


def univariate_lipid_regression(multimodal_df):
    """
    Perform univariate regression analysis for lipid intensity values against
    the probability of class 5. This function iterates through all lipid columns in the
    provided multimodal DataFrame, fits a GLM model for each, and returns a DataFrame
    with coefficients, p-values, and FDR-corrected p-values.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
    Returns:
        top20 (pd.DataFrame): DataFrame with the top 20 lipids based on FDR-corrected
        p-values.
        results_df (pd.DataFrame): DataFrame with lipid names as index, coefficients,
        p-values, and FDR-corrected p-values.

    """
    lipid_columns = [col for col in multimodal_df.columns if col.startswith("gpeak")]
    records = []
    for lipid in lipid_columns:
        subset = multimodal_df[
            [lipid, "prob_class_5", "age", "sex", "bmi", "duration_illness", "smoker"]
        ].dropna()

        formula = f"prob_class_5 ~ {lipid} + age + C(sex) + bmi + duration_illness + "
        "C(smoker)"

        model = smf.glm(formula, data=subset)
        result = model.fit(cov_type="HC3")  # adjust for heteroscedasticity
        coef = result.params.get(lipid, np.nan)
        se = result.bse.get(lipid, np.nan)
        pval = result.pvalues.get(lipid, np.nan)

        # 95% CI (respects robust covariance)
        ci_low, ci_high = result.conf_int().loc[lipid].tolist()

        records.append(
            {
                "lipid": lipid,
                "coef": coef,
                "pval": pval,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    results_df = pd.DataFrame(records).set_index("lipid")
    results_df["FDR"] = multipletests(results_df["pval"], method="fdr_bh")[1]
    results_df["log10_FDR"] = -np.log10(results_df["FDR"])

    top20 = results_df.nsmallest(20, "FDR")
    results_df["log10_FDR"] = -np.log10(
        np.clip(results_df["FDR"], np.finfo(float).tiny, None)
    )

    return top20, results_df


def univariate_lipid_class_regression(multimodal_df):
    """
    Perform univariate regression analysis for PRS (Polygenic Risk Scores) against
    the probability of class 5. This function iterates through all PRS columns in the
    provided multimodal DataFrame, fits a GLM model for each, and returns a DataFrame
    with coefficients, p-values, and FDR-corrected p-values.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
    Returns:
        pd.DataFrame: DataFrame with PRS names as index, coefficients, p-values,
        and FDR-corrected p-values.
    """

    lipid_class_cols = [col for col in multimodal_df.columns if col.endswith("mean")]

    lipid_class_records = []
    for lipid_class in lipid_class_cols:
        subset = multimodal_df[
            [lipid_class, "prob_class_5", "age", "sex", "bmi"]
        ].dropna()

        formula = f"prob_class_5 ~ {lipid_class} + age + C(sex) + bmi "
        model = smf.glm(formula, data=subset).fit()
        coef = model.params.get(lipid_class, np.nan)
        pval = model.pvalues.get(lipid_class, np.nan)
        lipid_class_records.append(
            {"lipid_class": lipid_class, "coef": coef, "pval": pval}
        )

    results_df = pd.DataFrame(lipid_class_records).set_index("lipid_class")
    results_df["FDR"] = multipletests(results_df["pval"], method="fdr_bh")[1]
    results_df["log10_FDR"] = -np.log10(results_df["FDR"])

    return results_df


def univariate_lipid_regression_cov_diag(multimodal_df):
    """
    Perform univariate regression analysis for lipid intensity values against
    the probability of class 5. This function iterates through all lipid columns in the
    provided multimodal DataFrame, fits a GLM model for each, and returns a DataFrame
    with coefficients, p-values, and FDR-corrected p-values.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
    Returns:
        top20 (pd.DataFrame): DataFrame with the top 20 lipids based on FDR-corrected
        p-values.
        results_df (pd.DataFrame): DataFrame with lipid names as index, coefficients,
        p-values, and FDR-corrected p-values.

    """
    lipid_columns = [col for col in multimodal_df.columns if col.startswith("gpeak")]
    records = []
    for lipid in lipid_columns:
        subset = multimodal_df[[lipid, "prob_class_5", "age", "sex", "bmi"]].dropna()

        formula = f"prob_class_5 ~ {lipid} + age + C(sex) + bmi"
        model = smf.glm(formula, data=subset).fit()
        coef = model.params.get(lipid, np.nan)
        pval = model.pvalues.get(lipid, np.nan)

        records.append({"lipid": lipid, "coef": coef, "pval": pval})

    results_df_cov_diag = pd.DataFrame(records).set_index("lipid")
    results_df_cov_diag["FDR"] = multipletests(
        results_df_cov_diag["pval"], method="fdr_bh"
    )[1]
    results_df_cov_diag["log10_FDR"] = -np.log10(results_df_cov_diag["FDR"])

    top20_cov_diag = results_df_cov_diag.nsmallest(20, "FDR")

    return top20_cov_diag, results_df_cov_diag


def univariate_lipids_ancova(multimodal_df):
    """
    Perform univariate ANCOVA analysis for lipid intensity values against
    the probability of class 5, focusing on the top and bottom 50 extreme values.
    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
    Returns:
        pd.DataFrame: DataFrame with lipid names as index, coefficients, p-values,
        and FDR-corrected p-values.
        top20 (pd.DataFrame): DataFrame with the top 20 lipids based on FDR-corrected
        p-values.
    """

    clean_df = multimodal_df.dropna()

    thresholds = [50, 100, 120]
    lipid_columns = [col for col in multimodal_df.columns if col.startswith("gpeak")]
    results_dfs = {}
    top20_dfs = {}
    for threshold in thresholds:
        top_idx = clean_df["prob_class_5"].nlargest(threshold).index
        bottom_idx = clean_df["prob_class_5"].nsmallest(threshold).index
        extreme_df = clean_df.loc[top_idx.union(bottom_idx)].copy()
        extreme_df["extreme_group"] = 0  # default: bottom
        extreme_df.loc[top_idx, "extreme_group"] = 1  # mark top
        lipid_records = []
        subset_dfs = []

        for lipid in lipid_columns:
            subset = extreme_df[
                [
                    lipid,
                    "prob_class_5",
                    "extreme_group",
                    "age",
                    "sex",
                    "bmi",
                    "diagnosis",
                ]
            ].dropna()
            formula = f"{lipid} ~ C(extreme_group) + age + C(sex) + bmi"
            model = smf.ols(formula, data=subset).fit()
            group_coef = model.params.get("C(extreme_group)[T.1]", np.nan)
            group_pval = model.pvalues.get("C(extreme_group)[T.1]", np.nan)

            lipid_records.append(
                {"lipid": lipid, "coef": group_coef, "pval": group_pval}
            )
            subset_dfs.append(subset)

        results_df = pd.DataFrame(lipid_records).set_index("lipid")
        results_df["FDR"] = multipletests(results_df["pval"], method="fdr_bh")[1]
        results_df["log10_FDR"] = -np.log10(results_df["FDR"])
        results_df = results_df.sort_values(by="FDR")
        results_dfs[threshold] = results_df

        top20 = results_df.nsmallest(20, "FDR")
        top20_dfs[threshold] = top20

    return results_dfs, top20_dfs


def lipid_class_enrichment_perm(
    results_df,
    annot_df,
    score_col="pval",
    n_perms=5000,
    random_state=42,
    min_in_class=5,
):
    """
    Permutation-based enrichment test for lipid classes (GSEA-style).

    results_df: DataFrame with regression results (index=lipid, must have `score_col`)
    annot_df:   DataFrame with lipid -> class mapping (index=lipid, col='class')
    score_col:  Which column to rank by (default: 'pval')
    n_perms:    Number of permutations
    """
    rng = np.random.default_rng(random_state)

    # merge results + annotations
    merged = results_df[[score_col]].join(annot_df[["class"]], how="inner").dropna()
    if merged.empty:
        return pd.DataFrame()

    # rank lipids: smaller p = stronger -> higher score
    scores = -np.log10(merged[score_col].clip(lower=np.finfo(float).tiny))
    merged["rank"] = scores.rank(ascending=False, method="average").to_numpy()

    all_ranks = merged["rank"].to_numpy()
    all_classes = merged["class"].to_numpy()
    # lipids = merged.index.to_numpy()

    rows = []
    for cl in np.unique(all_classes):
        mask = all_classes == cl
        n = mask.sum()
        if n < min_in_class:
            continue

        # observed enrichment score (mean rank difference)
        obs_es = merged.loc[mask, "rank"].mean() - merged.loc[~mask, "rank"].mean()

        # permutation null distribution
        null_es = []
        for _ in range(n_perms):
            perm_mask = rng.choice(len(all_ranks), size=n, replace=False)
            null_es.append(
                all_ranks[perm_mask].mean()
                - all_ranks[~np.isin(np.arange(len(all_ranks)), perm_mask)].mean()
            )
        null_es = np.array(null_es)

        # one-sided p-value: is observed > expected?
        pval = (np.sum(null_es >= obs_es) + 1) / (n_perms + 1)

        rows.append(
            {
                "class": cl,
                "n_in_class": n,
                "ES": obs_es,
                "pval": pval,
            }
        )

    out = pd.DataFrame(rows).set_index("class")
    if out.empty:
        return out

    out["FDR"] = multipletests(out["pval"], method="fdr_bh")[1]
    out["-log10(FDR)"] = -np.log10(np.clip(out["FDR"], np.finfo(float).tiny, None))
    return out.sort_values("FDR")
