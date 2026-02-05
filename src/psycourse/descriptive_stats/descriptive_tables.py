import numpy as np
import pandas as pd
from scipy import stats


def get_demographics_table(df_full, df_subset=None, group_diagnoses=True):
    """Generate comprehensive demographics table with optional comparison.

    Args:
        df_full: Full cohort dataframe
        df_subset: Optional subset dataframe (e.g., lipid subset) for comparison
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

    def t_test_p(series1, series2):
        """Two-sample t-test p-value."""
        try:
            s1 = series1.dropna()
            s2 = series2.dropna()
            if len(s1) < 2 or len(s2) < 2:
                return np.nan
            _, p = stats.ttest_ind(s1, s2, equal_var=False)
            return p
        except Exception:
            return np.nan

    def chi2_test_p(series1, series2):
        """Chi-square test p-value for categorical variables."""
        try:
            contingency = pd.crosstab(
                series1.fillna("Missing"), series2.fillna("Missing")
            )
            if contingency.size == 0:
                return np.nan
            # Use Fisher's exact for small cell counts
            if (contingency < 5).any().any():
                _, p = (
                    stats.fisher_exact(contingency)
                    if contingency.shape == (2, 2)
                    else (np.nan, np.nan)
                )
            else:
                _, p, _, _ = stats.chi2_contingency(contingency)
            return p
        except Exception:
            return np.nan

    def mann_whitney_p(series1, series2):
        """Mann-Whitney U test for non-normal continuous variables."""
        try:
            s1 = series1.dropna()
            s2 = series2.dropna()
            if len(s1) < 2 or len(s2) < 2:
                return np.nan
            _, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            return p
        except Exception:
            return np.nan

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
        df_full = df_full.copy()
        df_full["diagnosis_grouped"] = df_full["diagnosis_sum"].map(diagnosis_map)
        diagnosis_col = "diagnosis_grouped"
        if df_subset is not None:
            df_subset = df_subset.copy()
            df_subset["diagnosis_grouped"] = df_subset["diagnosis_sum"].map(
                diagnosis_map
            )
    else:
        diagnosis_col = "diagnosis_sum"

    rows = []

    # Determine if we're doing comparison
    comparing = df_subset is not None

    # === SAMPLE SIZE ===
    row = {"Characteristic": "N"}
    row["All Participants"] = f"{len(df_full)}"
    if comparing:
        row["Lipid Subset"] = f"{len(df_subset)}"
        row["p-value"] = "—"
    rows.append(row)

    # === SECTION: DEMOGRAPHICS ===
    rows.append(
        {
            "Characteristic": "**Demographics**",
            "All Participants": "",
            "Lipid Subset": "" if comparing else None,
            "p-value": "" if comparing else None,
        }
    )

    # Age
    row = {"Characteristic": "Age, mean (SD), years"}
    row["All Participants"] = format_mean_sd(df_full["age"])
    if comparing:
        row["Lipid Subset"] = format_mean_sd(df_subset["age"])
        row["p-value"] = f"{t_test_p(df_full['age'], df_subset['age']):.3f}"
    rows.append(row)

    # Sex
    if "sex" in df_full.columns:
        if comparing:
            df_full_temp = df_full.copy()
            df_subset_temp = df_subset.copy()
            df_full_temp["subset"] = "All"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([df_full_temp, df_subset_temp])
            sex_p = chi2_test_p(combined["sex"], combined["subset"])

        row = {"Characteristic": "Sex, n (%)"}
        row["All Participants"] = ""
        if comparing:
            row["Lipid Subset"] = ""
            row["p-value"] = f"{sex_p:.3f}" if not np.isnan(sex_p) else "—"
        rows.append(row)

        for sex in ["M", "F"]:
            row = {"Characteristic": f'  {sex}{"ale" if sex == "M" else "emale"}'}
            row["All Participants"] = format_count_pct(
                df_full["sex"], sex, len(df_full)
            )
            if comparing:
                row["Lipid Subset"] = format_count_pct(
                    df_subset["sex"], sex, len(df_subset)
                )
                row["p-value"] = ""
            rows.append(row)

    # BMI
    if "bmi" in df_full.columns:
        row = {"Characteristic": "BMI, mean (SD), kg/m²"}
        row["All Participants"] = format_mean_sd(df_full["bmi"])
        if comparing:
            row["Lipid Subset"] = format_mean_sd(df_subset["bmi"])
            row["p-value"] = f"{t_test_p(df_full['bmi'], df_subset['bmi']):.3f}"
        rows.append(row)

    # === SECTION: DIAGNOSIS ===
    rows.append(
        {
            "Characteristic": "**Diagnosis, n (%)**",
            "All Participants": "",
            "Lipid Subset": "" if comparing else None,
            "p-value": "" if comparing else None,
        }
    )

    if diagnosis_col in df_full.columns:
        if comparing:
            df_full_temp = df_full.copy()
            df_subset_temp = df_subset.copy()
            df_full_temp["subset"] = "All"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([df_full_temp, df_subset_temp])
            diag_p = chi2_test_p(combined[diagnosis_col], combined["subset"])

        # Add p-value to section header
        if comparing:
            rows[-1]["p-value"] = f"{diag_p:.3f}" if not np.isnan(diag_p) else "—"

        diagnoses = sorted(df_full[diagnosis_col].dropna().unique())
        for diag in diagnoses:
            row = {"Characteristic": f"  {diag}"}
            row["All Participants"] = format_count_pct(
                df_full[diagnosis_col], diag, len(df_full)
            )
            if comparing:
                row["Lipid Subset"] = format_count_pct(
                    df_subset[diagnosis_col], diag, len(df_subset)
                )
                row["p-value"] = ""
            rows.append(row)

    # === SECTION: CLINICAL VARIABLES ===
    rows.append(
        {
            "Characteristic": "**Clinical Variables**",
            "All Participants": "",
            "Lipid Subset": "" if comparing else None,
            "p-value": "" if comparing else None,
        }
    )

    # Duration of illness
    if "duration_illness" in df_full.columns:
        row = {"Characteristic": "Duration of illness, mean (SD), years"}
        row["All Participants"] = format_mean_sd(df_full["duration_illness"])
        if comparing:
            row["Lipid Subset"] = format_mean_sd(df_subset["duration_illness"])
            row["p-value"] = f"{t_test_p(df_full['duration_illness'],
                            df_subset['duration_illness']):.3f}"
        rows.append(row)

    # === SECTION: LIFESTYLE FACTORS ===
    rows.append(
        {
            "Characteristic": "**Lifestyle Factors**",
            "All Participants": "",
            "Lipid Subset": "" if comparing else None,
            "p-value": "" if comparing else None,
        }
    )

    # Smoking status
    if "smoker" in df_full.columns:
        if comparing:
            df_full_temp = df_full.copy()
            df_subset_temp = df_subset.copy()
            df_full_temp["subset"] = "All"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([df_full_temp, df_subset_temp])
            smoking_p = chi2_test_p(combined["smoker"], combined["subset"])

        row = {"Characteristic": "Smoking status, n (%)"}
        row["All Participants"] = ""
        if comparing:
            row["Lipid Subset"] = ""
            row["p-value"] = f"{smoking_p:.3f}" if not np.isnan(smoking_p) else "—"
        rows.append(row)

        for status in ["yes", "former", "never"]:
            if status in df_full["smoker"].values:
                row = {
                    "Characteristic": f'  {"Current" if status ==
                                           "yes" else status.capitalize()}'
                }
                row["All Participants"] = format_count_pct(
                    df_full["smoker"], status, len(df_full)
                )
                if comparing:
                    row["Lipid Subset"] = format_count_pct(
                        df_subset["smoker"], status, len(df_subset)
                    )
                    row["p-value"] = ""
                rows.append(row)

    # === SECTION: MEDICATION ===
    rows.append(
        {
            "Characteristic": "**Medication**",
            "All Participants": "",
            "Lipid Subset": "" if comparing else None,
            "p-value": "" if comparing else None,
        }
    )

    # Antipsychotic use (any use: count > 0)
    if "antipsychotics_count" in df_full.columns:
        df_full_temp = df_full.copy()
        df_full_temp["any_antipsychotic"] = df_full_temp["antipsychotics_count"] > 0

        row = {"Characteristic": "Antipsychotic use, n (%)"}
        row["All Participants"] = format_count_pct(
            df_full_temp["any_antipsychotic"], True, len(df_full)
        )

        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["any_antipsychotic"] = (
                df_subset_temp["antipsychotics_count"] > 0
            )
            row["Lipid Subset"] = format_count_pct(
                df_subset_temp["any_antipsychotic"], True, len(df_subset)
            )

            df_full_temp["subset"] = "All"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([df_full_temp, df_subset_temp])
            row["p-value"] = (
                f"{chi2_test_p(combined['any_antipsychotic'], combined['subset']):.3f}"
            )
        rows.append(row)

    # Mood stabilizer use (any use: count > 0)
    if "mood_stabilizers_count" in df_full.columns:
        df_full_temp = df_full.copy()
        df_full_temp["any_mood_stabilizer"] = df_full_temp["mood_stabilizers_count"] > 0

        row = {"Characteristic": "Mood stabilizer use, n (%)"}
        row["All Participants"] = format_count_pct(
            df_full_temp["any_mood_stabilizer"], True, len(df_full)
        )

        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["any_mood_stabilizer"] = (
                df_subset_temp["mood_stabilizers_count"] > 0
            )
            row["Lipid Subset"] = format_count_pct(
                df_subset_temp["any_mood_stabilizer"], True, len(df_subset)
            )

            df_full_temp["subset"] = "All"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([df_full_temp, df_subset_temp])
            row["p-value"] = f"{chi2_test_p(combined['any_mood_stabilizer'],
                               combined['subset']):.3f}"
        rows.append(row)

    # Antidepressant use (any use: count > 0)
    if "antidepressants_count" in df_full.columns:
        df_full_temp = df_full.copy()
        df_full_temp["any_antidepressant"] = df_full_temp["antidepressants_count"] > 0

        row = {"Characteristic": "Antidepressant use, n (%)"}
        row["All Participants"] = format_count_pct(
            df_full_temp["any_antidepressant"], True, len(df_full)
        )

        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["any_antidepressant"] = (
                df_subset_temp["antidepressants_count"] > 0
            )
            row["Lipid Subset"] = format_count_pct(
                df_subset_temp["any_antidepressant"], True, len(df_subset)
            )

            df_full_temp["subset"] = "All"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([df_full_temp, df_subset_temp])
            row["p-value"] = (
                f"{chi2_test_p(combined['any_antidepressant'], combined['subset']):.3f}"
            )
        rows.append(row)

    # Total medication count (sum of all medication classes)
    medication_cols = [
        "antipsychotics_count",
        "mood_stabilizers_count",
        "antidepressants_count",
        "tranquilizers_count",
    ]
    if all(col in df_full.columns for col in medication_cols):
        df_full_temp = df_full.copy()
        df_full_temp["total_medications"] = df_full[medication_cols].sum(axis=1)

        row = {"Characteristic": "Total medications, mean (SD)"}
        row["All Participants"] = format_mean_sd(df_full_temp["total_medications"])

        if comparing:
            df_subset_temp = df_subset.copy()
            df_subset_temp["total_medications"] = df_subset[medication_cols].sum(axis=1)
            row["Lipid Subset"] = format_mean_sd(df_subset_temp["total_medications"])
            row["p-value"] = f"{t_test_p(df_full_temp['total_medications'],
                            df_subset_temp['total_medications']):.3f}"
        rows.append(row)

    # === SECTION: SEVERE PSYCHOSIS SUBTYPE ===
    rows.append(
        {
            "Characteristic": "**Severe Psychosis Subtype**",
            "All Participants": "",
            "Lipid Subset": "" if comparing else None,
            "p-value": "" if comparing else None,
        }
    )

    # Cluster 5 probability
    if "prob_class_5" in df_full.columns:
        row = {"Characteristic": "Cluster 5 probability, mean (SD)"}
        row["All Participants"] = (
            f"{df_full['prob_class_5'].mean():.3f} "
            f"({df_full['prob_class_5'].std():.3f})"
        )
        if comparing:
            row["Lipid Subset"] = (
                f"{df_subset['prob_class_5'].mean():.3f} "
                f"({df_subset['prob_class_5'].std():.3f})"
            )
            row["p-value"] = (
                f"{t_test_p(df_full['prob_class_5'], df_subset['prob_class_5']):.3f}"
            )
        rows.append(row)

    # Cluster assignments
    if "predicted_label" in df_full.columns:
        if comparing:
            df_full_temp = df_full.copy()
            df_subset_temp = df_subset.copy()
            df_full_temp["subset"] = "All"
            df_subset_temp["subset"] = "Lipid"
            combined = pd.concat([df_full_temp, df_subset_temp])
            cluster_p = chi2_test_p(combined["predicted_label"], combined["subset"])

        row = {"Characteristic": "Cluster assignment, n (%)"}
        row["All Participants"] = ""
        if comparing:
            row["Lipid Subset"] = ""
            row["p-value"] = f"{cluster_p:.3f}" if not np.isnan(cluster_p) else "—"
        rows.append(row)

        for cluster in sorted(df_full["predicted_label"].dropna().unique()):
            label = f"Cluster {cluster}" + (" (Severe)" if cluster == 5 else "")
            row = {"Characteristic": f"  {label}"}
            row["All Participants"] = format_count_pct(
                df_full["predicted_label"], cluster, len(df_full)
            )
            if comparing:
                row["Lipid Subset"] = format_count_pct(
                    df_subset["predicted_label"], cluster, len(df_subset)
                )
                row["p-value"] = ""
            rows.append(row)

    # Create DataFrame
    result_df = pd.DataFrame(rows)

    # Clean up p-values (remove 'nan' strings, format)
    if comparing and "p-value" in result_df.columns:
        result_df["p-value"] = result_df["p-value"].replace("nan", "—")

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
