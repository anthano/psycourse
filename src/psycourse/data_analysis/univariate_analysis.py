import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

############## PRS ################


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
    results_df["log10_FDR"] = -np.log10(results_df["FDR"])

    return results_df


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
        subset = multimodal_df[[lipid, "prob_class_5", "age", "sex", "bmi"]].dropna()

        formula = f"prob_class_5 ~ {lipid} + age + C(sex) + bmi"
        model = smf.glm(formula, data=subset).fit()
        coef = model.params.get(lipid, np.nan)
        pval = model.pvalues.get(lipid, np.nan)

        records.append({"lipid": lipid, "coef": coef, "pval": pval})

    results_df = pd.DataFrame(records).set_index("lipid")
    results_df["FDR"] = multipletests(results_df["pval"], method="fdr_bh")[1]
    results_df["log10_FDR"] = -np.log10(results_df["FDR"])

    top20 = results_df.nsmallest(20, "FDR")
    results_df["log10_FDR"] = -np.log10(results_df["FDR"])

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
