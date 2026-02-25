import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

########################################################################################
# PRS
########################################################################################

# PRS standard covs = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]
# PRS sensitivity = ["bmi"], ["diagnosis"]


def univariate_prs_regression_panss(multimodal_df, panss_column):
    """
    Univariate GLMs of panss_column on each PRS, adjusted for covariates.
    Returns coef, SE, 95% CI, p, FDR, and -log10(FDR).
    Uses HC3 robust covariance (with a safe fallback for older statsmodels).
    Args:
        multimodal_df (pd.DataFrame): DataFrame containing PRS and covariates.
        panss_column (str): The name of the PANSS column to use in the regression.
    Returns:
        df (pd.DataFrame): pd.DataFrame with results for every PRS.
        n_subset_dict (dict): contains the n numbers per PRS.
    """
    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]
    rows = []
    n_subset_dict = {}

    for prs in prs_columns:
        cols = [
            prs,
            panss_column,
            "age",
            "sex",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
        ]
        subset = multimodal_df[cols].dropna()
        n_subset = len(subset)
        n_subset_dict[prs] = n_subset

        formula = (
            f"{panss_column} ~ {prs} + age + C(sex)  + " "pc1 + pc2 + pc3 + pc4 + pc5"
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
    return df, n_subset_dict


def univariate_prs_regression_cov_bmi_panss(multimodal_df, panss_column):
    """
    Univariate GLMs of panss_column on each PRS, adjusted for covariates
    - added bmi for sensitivity.
    Returns coef, SE, 95% CI, p, FDR, and -log10(FDR).
    Uses HC3 robust covariance (with a safe fallback for older statsmodels).
    Args:
        multimodal_df (pd.DataFrame): DataFrame containing PRS and covariates.
        panss_column (str): The name of the PANSS column to use in the regression.
    Returns:
        df (pd.DataFrame): pd.DataFrame with results for every PRS.
        n_subset_dict (dict): contains the n numbers per PRS.
    """
    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]
    rows = []
    n_subset_dict = {}

    for prs in prs_columns:
        cols = [
            prs,
            panss_column,
            "age",
            "sex",
            "bmi",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
        ]
        subset = multimodal_df[cols].dropna()
        n_subset_dict[prs] = len(subset)

        formula = (
            f"{panss_column} ~ {prs} + age + C(sex) + bmi  + "
            "pc1 + pc2 + pc3 + pc4 + pc5"
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
    return df, n_subset_dict


def univariate_prs_regression_cov_diagnosis_panss(multimodal_df, panss_column):
    """
    Univariate GLMs of panss_column on each PRS, adjusted for covariates
    - added bmi for sensitivity.
    Returns coef, SE, 95% CI, p, FDR, and -log10(FDR).
    Uses HC3 robust covariance (with a safe fallback for older statsmodels).
    Args:
        multimodal_df (pd.DataFrame): DataFrame containing PRS and covariates.
        panss_column (str): The name of the PANSS column to use in the regression.
    Returns:
        df (pd.DataFrame): pd.DataFrame with results for every PRS.
        n_subset_dict (dict): contains the n numbers per PRS.
    """
    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]
    rows = []
    n_subset_dict = {}

    for prs in prs_columns:
        cols = [
            prs,
            panss_column,
            "age",
            "sex",
            "diagnosis_sum",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
            "pc5",
        ]
        subset = multimodal_df[cols].dropna()
        n_subset = len(subset)
        n_subset_dict[prs] = n_subset

        formula = (
            f"{panss_column} ~ {prs} + age + C(sex) + C(diagnosis_sum)  + "
            "pc1 + pc2 + pc3 + pc4 + pc5"
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
    return df, n_subset_dict


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
