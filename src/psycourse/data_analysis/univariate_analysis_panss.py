import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

########################################################################################
# Lipids
########################################################################################

# Standard Set of Covariates = ["age", "sex", "bmi", "duration_illness", "smoker"]
# Added Covariates = ["diagnosis_sum"]


def univariate_lipid_regression_panss(multimodal_df, panss_column):
    """
    Perform univariate regression analysis for lipid intensity values against
    the specified PANSS column. This function iterates through all lipid columns in the
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
    n_subset_lipid_dict = {}
    for lipid in lipid_columns:
        subset = multimodal_df[
            [lipid, panss_column, "age", "sex", "bmi", "duration_illness", "smoker"]
        ].dropna()
        n_subset_lipid_dict[lipid] = len(subset)

        formula = (
            f"{panss_column} ~ {lipid} + age + C(sex) + bmi + "
            "duration_illness + C(smoker)"
        )

        model = smf.ols(formula, data=subset)
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

    return n_subset_lipid_dict, top20, results_df


def univariate_lipid_regression_cov_diagnosis_panss(multimodal_df, panss_column):
    """
    Perform univariate regression analysis for lipid intensity values against
    the specified PANSS column with an added covariate diagnoses.
    This function iterates through all lipid columns in the
    provided multimodal DataFrame, fits a GLM model for each, and returns a DataFrame
    with coefficients, p-values, and FDR-corrected p-values.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
        panss_column (str): The name of the PANSS column to use in the regression.
    Returns:
        top20 (pd.DataFrame): DataFrame with the top 20 lipids based on FDR-corrected
        p-values.
        results_df (pd.DataFrame): DataFrame with lipid names as index, coefficients,
        p-values, and FDR-corrected p-values.

    """
    lipid_columns = [col for col in multimodal_df.columns if col.startswith("gpeak")]
    records = []
    n_subset_lipid_dict = {}
    for lipid in lipid_columns:
        subset = multimodal_df[
            [
                lipid,
                panss_column,
                "age",
                "sex",
                "bmi",
                "duration_illness",
                "smoker",
                "diagnosis_sum",
            ]
        ].dropna()

        n_subset_lipid_dict[lipid] = len(subset)

        formula = (
            f"{panss_column} ~ {lipid} + age + C(sex) + bmi + "
            "duration_illness + C(smoker) + C(diagnosis_sum)"
        )

        model = smf.ols(formula, data=subset)
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

    return n_subset_lipid_dict, top20, results_df


def univariate_lipid_regression_cov_med_panss(multimodal_df, panss_column):
    """
    Perform univariate regression analysis for lipid intensity values against
    the panss score with an added covariate medication.
    This function iterates through all lipid columns in the
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
    n_subset_lipid_dict = {}
    for lipid in lipid_columns:
        subset = multimodal_df[
            [
                lipid,
                panss_column,
                "age",
                "sex",
                "bmi",
                "duration_illness",
                "smoker",
                "antidepressants_count",
                "antipsychotics_count",
                "tranquilizers_count",
                "mood_stabilizers_count",
            ]
        ].dropna()

        n_subset_lipid_dict[lipid] = len(subset)

        formula = (
            f"{panss_column} ~ {lipid} + age + C(sex) + bmi + "
            "duration_illness + C(smoker) + antidepressants_count + "
            "antipsychotics_count + tranquilizers_count + mood_stabilizers_count"
        )

        model = smf.ols(formula, data=subset)
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

    return n_subset_lipid_dict, top20, results_df


def univariate_lipid_regression_cov_med_and_diag_panss(multimodal_df, panss_column):
    """
    Perform univariate regression analysis for lipid intensity values against
    the specified PANSS column with an added covariate medication + diagnosis.
    This function iterates through all lipid columns in the
    provided multimodal DataFrame, fits a GLM model for each, and returns a DataFrame
    with coefficients, p-values, and FDR-corrected p-values.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing multimodal data.
        panss_column (str): The name of the PANSS column to use in the regression.
    Returns:
        top20 (pd.DataFrame): DataFrame with the top 20 lipids based on FDR-corrected
        p-values.
        results_df (pd.DataFrame): DataFrame with lipid names as index, coefficients,
        p-values, and FDR-corrected p-values.

    """
    lipid_columns = [col for col in multimodal_df.columns if col.startswith("gpeak")]
    records = []
    n_subset_lipid_dict = {}
    for lipid in lipid_columns:
        subset = multimodal_df[
            [
                lipid,
                panss_column,
                "age",
                "sex",
                "bmi",
                "duration_illness",
                "smoker",
                "diagnosis_sum",
                "antidepressants_count",
                "antipsychotics_count",
                "tranquilizers_count",
                "mood_stabilizers_count",
            ]
        ].dropna()

        n_subset_lipid_dict[lipid] = len(subset)

        formula = (
            f"{panss_column} ~ {lipid} + age + C(sex) + bmi + "
            "duration_illness + C(smoker) + C(diagnosis_sum) + antidepressants_count "
            "+ "
            "antipsychotics_count + tranquilizers_count + mood_stabilizers_count"
        )

        model = smf.ols(formula, data=subset)
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

    return n_subset_lipid_dict, top20, results_df
