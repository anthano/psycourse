import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ======================================================================================
# PRS
# ======================================================================================


def univariate_prs_ancova(multimodal_df):
    """
    Univariate PRS ANCOVA with binarized predicted labels (G5 vs. others).

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing PRS, predicted labels,
        and covariates.

    Returns:
        pd.DataFrame: ANCOVA results for each PRS with coefficients, standard errors,
        confidence intervals, p-values, and FDR-corrected p-values.
    """

    df = multimodal_df.copy()
    df["binarized_label"] = (multimodal_df["predicted_label"] == 5).astype(int)

    prs_columns = [c for c in df.columns if c.endswith("PRS")]
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

    rows = []
    for prs in prs_columns:
        subset_cols = [prs, "binarized_label"] + covars
        sub = df[subset_cols].dropna()

        # PRS ~ G5 + covariates
        formula = (
            f"{prs} ~ binarized_label + age + C(sex) + bmi + "
            "pc1 + pc2 + pc3 + pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10"
        )

        # OLS with HC3 robust covariance (matches your previous workflow)
        fit = smf.ols(formula=formula, data=sub).fit(cov_type="HC3")

        beta = fit.params.get("binarized_label", np.nan)
        se = fit.bse.get("binarized_label", np.nan)
        pval = fit.pvalues.get("binarized_label", np.nan)
        ci_l, ci_u = fit.conf_int().loc["binarized_label"].tolist()

        rows.append(
            {
                "prs": prs,
                "N": len(sub),
                "coef": beta,
                "se": se,
                "ci_low": ci_l,
                "ci_high": ci_u,
                "pval": pval,
            }
        )

    out = pd.DataFrame(rows).set_index("prs")
    out["FDR"] = multipletests(out["pval"].fillna(1.0), method="fdr_bh")[1]
    out["log10_FDR"] = -np.log10(np.clip(out["FDR"], np.finfo(float).tiny, None))
    out = out.sort_values("FDR")
    return out
