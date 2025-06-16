import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def univariate_prs_regression(multimodal_df):
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

    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]
    print(prs_columns)

    prs_records = []
    for prs in prs_columns:
        subset = multimodal_df[[prs, "prob_class_5", "age", "sex", "bmi"]].dropna()

        formula = f"prob_class_5 ~ {prs} + age + C(sex) + bmi "
        model = smf.glm(formula, data=subset).fit()
        coef = model.params.get(prs, np.nan)
        pval = model.pvalues.get(prs, np.nan)
        prs_records.append({"prs": prs, "coef": coef, "pval": pval})

    results_df = pd.DataFrame(prs_records).set_index("prs")
    results_df["FDR"] = multipletests(results_df["pval"], method="fdr_bh")[1]

    return results_df


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
        subset = multimodal_df[[lipid, "prob_class_5", "age", "sex", "bmi"]].dropna()

        formula = f"prob_class_5 ~ {lipid} + age + C(sex) + bmi"
        model = smf.glm(formula, data=subset).fit()
        coef = model.params.get(lipid, np.nan)
        pval = model.pvalues.get(lipid, np.nan)

        records.append({"lipid": lipid, "coef": coef, "pval": pval})

    results_df = pd.DataFrame(records).set_index("lipid")
    results_df["FDR"] = multipletests(results_df["pval"], method="fdr_bh")[1]

    top20 = results_df.nsmallest(20, "FDR")

    return top20, results_df
