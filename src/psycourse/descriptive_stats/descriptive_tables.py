import numpy as np
import pandas as pd
from scipy import stats


def get_demographics_table(df_full, df_subset=None, df_prs=None, group_diagnoses=True):
    """Generate comprehensive demographics table with optional comparison.

    Args:
        df_full: Full cohort dataframe
        df_subset: Optional subset dataframe (e.g., lipid subset) for comparison
        df_prs: Optional PRS subset dataframe for an additional column
        group_diagnoses: If True, collapse diagnoses into broader categories

    Returns:
        pd.DataFrame: Formatted demographics table with statistics
    """

    def format_mean_sd(series):
        """Format mean (SD) for continuous variables."""
        if series.isna().all():
            return "—"
        return f"{series.mean():.1f} ({series.std():.1f})"

    def format_median_iqr(series):
        """Format median (IQR) for skewed variables."""
        if series.isna().all():
            return "—"
        q25, q50, q75 = series.quantile([0.25, 0.5, 0.75])
        return f"{q50:.1f} ({q25:.1f}–{q75:.1f})"

    def format_count_pct(series, value, total):
        """Format n (%) for categorical variables."""
        count = (series == value).sum()
        pct = count / total * 100 if total > 0 else 0
        return f"{count} ({pct:.1f}%)"

    def t_test_stat_p(series1, series2):
        """Two-sample t-test, returns (t, p)."""
        try:
            s1 = series1.dropna()
            s2 = series2.dropna()
            if len(s1) < 2 or len(s2) < 2:
                return np.nan, np.nan
            t, p = stats.ttest_ind(s1, s2, equal_var=False)
            return t, p
        except Exception:
            return np.nan, np.nan

    def chi2_test_stat_p(series1, series2):
        """Chi-square test, returns (stat, p) for categorical variables."""
        try:
            contingency = pd.crosstab(
                series1.fillna("Missing"), series2.fillna("Missing")
            )
            if contingency.size == 0:
                return np.nan, np.nan
            # Use Fisher's exact for small cell counts
            if (contingency < 5).any().any():
                if contingency.shape == (2, 2):
                    _, p = stats.fisher_exact(contingency)
                    return np.nan, p  # Fisher exact has no single chi2 stat
                else:
                    return np.nan, np.nan
            else:
                chi2, p, _, _ = stats.chi2_contingency(contingency)
                return chi2, p
        except Exception:
            return np.nan, np.nan

    def mann_whitney_stat_p(series1, series2):
        """Mann-Whitney U test, returns (U, p)."""
        try:
            s1 = series1.dropna()
            s2 = series2.dropna()
            if len(s1) < 2 or len(s2) < 2:
                return np.nan, np.nan
            u, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            return u, p
        except Exception:
            return np.nan, np.nan

    def fmt_t(stat, p):
        """Format t-statistic and p-value."""
        if np.isnan(p):
            return "—"
        if np.isnan(stat):
            return f"p={p:.3f}"
        return f"t={stat:.2f}, p={p:.3f}"

    def fmt_chi2(stat, p):
        """Format chi-square (or Fisher) statistic and p-value."""
        if np.isnan(p):
            return "—"
        if np.isnan(stat):
            return f"p={p:.3f}"  # Fisher exact — no chi2 stat
        return f"χ²={stat:.2f}, p={p:.3f}"

    # Determine primary dataset and column name for the first data column.
    # If df_prs is provided it replaces df_full as the displayed primary column.
    if df_prs is not None:
        primary_df = df_prs.copy()
        primary_col = f"PRS sample (n={len(df_prs)})"
    else:
        primary_df = df_full.copy()
        primary_col = "All Participants"

    # Optionally group diagnoses into broader categories
    if group_diagnoses:
        diagnosis_map = {
            "Schizophrenia": "Schizophrenia spectrum",
            "ICD-10 Schizophrenia": "Schizophrenia spectrum",
            "Schizoaffective Disorder": "Schizophrenia spectrum",
            "Schizophreniform Disorder": "Schizophrenia spectrum",
            "Brief Psychotic Disorder": "Schizophrenia spectrum",
            "Bipolar-I Disorder": "Bipolar disorder",
            "Bipolar-II Disorder": "Bipolar disorder",
            "Recurrent Depression": "Recurrent depression",
        }
        primary_df["diagnosis_grouped"] = primary_df["diagnosis_sum"].map(diagnosis_map)
        diagnosis_col = "diagnosis_grouped"
        if df_subset is not None:
            df_subset = df_subset.copy()
            df_subset["diagnosis_grouped"] = df_subset["diagnosis_sum"].map(
                diagnosis_map
            )
    else:
        diagnosis_col = "diagnosis_sum"

    rows = []

    # Determine if we're doing a two-column comparison
    comparing = df_subset is not None

    def section_row(label):
        return {
            "Characteristic": label,
            primary_col: "",
            "Lipid Subset": "" if comparing else None,
            "p-value": "" if comparing else None,
        }

    # === SAMPLE SIZE ===
    row = {"Characteristic": "N", primary_col: f"{len(primary_df)}"}
    if comparing:
        row["Lipid Subset"] = f"{len(df_subset)}"
        row["p-value"] = "—"
    rows.append(row)

    # === SECTION: DEMOGRAPHICS ===
    rows.append(section_row("**Demographics**"))

    # Age
    row = {"Characteristic": "Age, mean (SD), years"}
    row[primary_col] = format_mean_sd(primary_df["age"])
    if comparing:
        row["Lipid Subset"] = format_mean_sd(df_subset["age"])
        t, p = t_test_stat_p(primary_df["age"], df_subset["age"])
        row["p-value"] = fmt_t(t, p)
    rows.append(row)

    # Sex
    if "sex" in primary_df.columns:
        if comparing:
            prim_temp = primary_df.copy()
            df_subset_temp = df_subset.copy()
            prim_temp["subset"] = "PRS"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([prim_temp, df_subset_temp])
            sex_chi2, sex_p = chi2_test_stat_p(combined["sex"], combined["subset"])

        row = {"Characteristic": "Sex, n (%)", primary_col: ""}
        if comparing:
            row["Lipid Subset"] = ""
            row["p-value"] = fmt_chi2(sex_chi2, sex_p)
        rows.append(row)

        for sex in ["M", "F"]:
            row = {"Characteristic": f'  {sex}{"ale" if sex == "M" else "emale"}'}
            row[primary_col] = format_count_pct(primary_df["sex"], sex, len(primary_df))
            if comparing:
                row["Lipid Subset"] = format_count_pct(
                    df_subset["sex"], sex, len(df_subset)
                )
                row["p-value"] = ""
            rows.append(row)

    # BMI
    if "bmi" in primary_df.columns:
        row = {"Characteristic": "BMI, mean (SD), kg/m²"}
        row[primary_col] = format_mean_sd(primary_df["bmi"])
        if comparing:
            row["Lipid Subset"] = format_mean_sd(df_subset["bmi"])
            t, p = t_test_stat_p(primary_df["bmi"], df_subset["bmi"])
            row["p-value"] = fmt_t(t, p)
        rows.append(row)

    # === SECTION: DIAGNOSIS ===
    rows.append(section_row("**Diagnosis, n (%)**"))

    if diagnosis_col in primary_df.columns:
        if comparing:
            prim_temp = primary_df.copy()
            df_subset_temp = df_subset.copy()
            prim_temp["subset"] = "PRS"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([prim_temp, df_subset_temp])
            diag_chi2, diag_p = chi2_test_stat_p(
                combined[diagnosis_col], combined["subset"]
            )
            rows[-1]["p-value"] = fmt_chi2(diag_chi2, diag_p)

        diagnoses = sorted(primary_df[diagnosis_col].dropna().unique())
        for diag in diagnoses:
            row = {"Characteristic": f"  {diag}"}
            row[primary_col] = format_count_pct(
                primary_df[diagnosis_col], diag, len(primary_df)
            )
            if comparing:
                row["Lipid Subset"] = format_count_pct(
                    df_subset[diagnosis_col], diag, len(df_subset)
                )
                row["p-value"] = ""
            rows.append(row)

    # === SECTION: CLINICAL VARIABLES ===
    rows.append(section_row("**Clinical Variables**"))

    # Duration of illness
    if "duration_illness" in primary_df.columns:
        row = {"Characteristic": "Duration of illness, mean (SD), years"}
        row[primary_col] = format_mean_sd(primary_df["duration_illness"])
        if comparing:
            row["Lipid Subset"] = format_mean_sd(df_subset["duration_illness"])
            t, p = t_test_stat_p(
                primary_df["duration_illness"], df_subset["duration_illness"]
            )
            row["p-value"] = fmt_t(t, p)
        rows.append(row)

    # === SECTION: LIFESTYLE FACTORS ===
    rows.append(section_row("**Lifestyle Factors**"))

    # Smoking status
    if "smoker" in primary_df.columns:
        if comparing:
            prim_temp = primary_df.copy()
            df_subset_temp = df_subset.copy()
            prim_temp["subset"] = "PRS"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([prim_temp, df_subset_temp])
            smoking_chi2, smoking_p = chi2_test_stat_p(
                combined["smoker"], combined["subset"]
            )

        row = {"Characteristic": "Smoking status, n (%)", primary_col: ""}
        if comparing:
            row["Lipid Subset"] = ""
            row["p-value"] = fmt_chi2(smoking_chi2, smoking_p)
        rows.append(row)

        for status in ["yes", "former", "never"]:
            if status in primary_df["smoker"].values:
                row = {
                    "Characteristic": f'  {"Current" if status == "yes"
                                           else status.capitalize()}'
                }
                row[primary_col] = format_count_pct(
                    primary_df["smoker"], status, len(primary_df)
                )
                if comparing:
                    row["Lipid Subset"] = format_count_pct(
                        df_subset["smoker"], status, len(df_subset)
                    )
                    row["p-value"] = ""
                rows.append(row)

    # === SECTION: MEDICATION ===
    rows.append(section_row("**Medication**"))

    # Antipsychotic use (any use: count > 0)
    if "antipsychotics_count" in primary_df.columns:
        prim_temp = primary_df.copy()
        prim_temp["any_antipsychotic"] = prim_temp["antipsychotics_count"] > 0

        row = {"Characteristic": "Antipsychotic use, n (%)"}
        row[primary_col] = format_count_pct(
            prim_temp["any_antipsychotic"], True, len(primary_df)
        )
        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["any_antipsychotic"] = (
                df_subset_temp["antipsychotics_count"] > 0
            )
            row["Lipid Subset"] = format_count_pct(
                df_subset_temp["any_antipsychotic"], True, len(df_subset)
            )
            prim_temp["subset"] = "PRS"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([prim_temp, df_subset_temp])
            chi2, p = chi2_test_stat_p(
                combined["any_antipsychotic"], combined["subset"]
            )
            row["p-value"] = fmt_chi2(chi2, p)
        rows.append(row)

    # Mood stabilizer use (any use: count > 0)
    if "mood_stabilizers_count" in primary_df.columns:
        prim_temp = primary_df.copy()
        prim_temp["any_mood_stabilizer"] = prim_temp["mood_stabilizers_count"] > 0

        row = {"Characteristic": "Mood stabilizer use, n (%)"}
        row[primary_col] = format_count_pct(
            prim_temp["any_mood_stabilizer"], True, len(primary_df)
        )
        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["any_mood_stabilizer"] = (
                df_subset_temp["mood_stabilizers_count"] > 0
            )
            row["Lipid Subset"] = format_count_pct(
                df_subset_temp["any_mood_stabilizer"], True, len(df_subset)
            )
            prim_temp["subset"] = "PRS"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([prim_temp, df_subset_temp])
            chi2, p = chi2_test_stat_p(
                combined["any_mood_stabilizer"], combined["subset"]
            )
            row["p-value"] = fmt_chi2(chi2, p)
        rows.append(row)

    # Antidepressant use (any use: count > 0)
    if "antidepressants_count" in primary_df.columns:
        prim_temp = primary_df.copy()
        prim_temp["any_antidepressant"] = prim_temp["antidepressants_count"] > 0

        row = {"Characteristic": "Antidepressant use, n (%)"}
        row[primary_col] = format_count_pct(
            prim_temp["any_antidepressant"], True, len(primary_df)
        )
        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["any_antidepressant"] = (
                df_subset_temp["antidepressants_count"] > 0
            )
            row["Lipid Subset"] = format_count_pct(
                df_subset_temp["any_antidepressant"], True, len(df_subset)
            )
            prim_temp["subset"] = "PRS"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([prim_temp, df_subset_temp])
            chi2, p = chi2_test_stat_p(
                combined["any_antidepressant"], combined["subset"]
            )
            row["p-value"] = fmt_chi2(chi2, p)
        rows.append(row)

    # Total medication count (sum of all medication classes)
    medication_cols = [
        "antipsychotics_count",
        "mood_stabilizers_count",
        "antidepressants_count",
        "tranquilizers_count",
    ]
    if all(col in primary_df.columns for col in medication_cols):
        prim_temp = primary_df.copy()
        prim_temp["total_medications"] = primary_df[medication_cols].sum(axis=1)

        row = {"Characteristic": "Total medications, mean (SD)"}
        row[primary_col] = format_mean_sd(prim_temp["total_medications"])
        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["total_medications"] = df_subset[medication_cols].sum(axis=1)
            row["Lipid Subset"] = format_mean_sd(df_subset_temp["total_medications"])
            t, p = t_test_stat_p(
                prim_temp["total_medications"], df_subset_temp["total_medications"]
            )
            row["p-value"] = fmt_t(t, p)
        rows.append(row)

    # === SECTION: SEVERE PSYCHOSIS SUBTYPE ===
    rows.append(section_row("**Severe Psychosis Subtype**"))

    # Severe psychosis subtype probability
    if "prob_class_5" in primary_df.columns:
        row = {"Characteristic": "Severe psychosis subtype probability, mean (SD)"}
        row[primary_col] = (
            f"{primary_df['prob_class_5'].mean():.3f} "
            f"({primary_df['prob_class_5'].std():.3f})"
        )
        if comparing:
            row["Lipid Subset"] = (
                f"{df_subset['prob_class_5'].mean():.3f} "
                f"({df_subset['prob_class_5'].std():.3f})"
            )
            t, p = t_test_stat_p(primary_df["prob_class_5"], df_subset["prob_class_5"])
            row["p-value"] = fmt_t(t, p)
        rows.append(row)

    # Create DataFrame
    result_df = pd.DataFrame(rows)

    # Drop any residual None-keyed columns
    if None in result_df.columns:
        result_df = result_df.drop(columns=[None])

    return result_df


########################################################################################
# Lipid Table
########################################################################################


def create_lipid_table(cleaned_lipid_class_data):
    """Already exists as pickle.
    Load it, fix index, export to csv.
    """
    lipid_df = cleaned_lipid_class_data.copy()
    # lipid_df = pd.read_pickle(BLD_DATA / "cleaned_lipid_class_data.pkl")
    lipid_df["lipid"] = lipid_df.index

    return lipid_df


def get_participants_per_analysis(
    standard_cov_dict,
    added_cov_1_dict,
    added_cov_2_dict,
    cov_1_name="added_cov_1",
    cov_2_name="added_cov_2",
):
    """Takes the n_subset dicts from the regression analysis and puts them into a df.

    Args:
        standard_cov_dict (dict): dict with predictor (PRS or lipid) as key and
            n of individuals as value.
        added_cov_1_dict (dict): dict with predictor as key and n of individuals
            as value (first additional covariate set).
        added_cov_2_dict (dict): dict with predictor as key and n of individuals
            as value (second additional covariate set).
        cov_1_name (str): Name for first additional covariate set column.
        cov_2_name (str): Name for second additional covariate set column.

    Returns:
        pd.DataFrame: Dataframe with predictors as rows and covariate sets as columns.
    """

    dicts_dict = {
        "standard_cov": standard_cov_dict,
        cov_1_name: added_cov_1_dict,
        cov_2_name: added_cov_2_dict,
    }

    n_per_analysis_df = pd.DataFrame(dicts_dict)

    return n_per_analysis_df
