import warnings

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
        """Two-sample t-test, returns (t, p).

        Converts to plain numpy float arrays so scipy doesn't choke on pandas
        nullable dtypes (Float64, etc.).
        """
        try:
            s1 = pd.to_numeric(series1, errors="coerce").dropna().to_numpy(dtype=float)
            s2 = pd.to_numeric(series2, errors="coerce").dropna().to_numpy(dtype=float)
            if len(s1) < 2 or len(s2) < 2:
                return np.nan, np.nan
            t, p = stats.ttest_ind(s1, s2, equal_var=False)
            return float(t), float(p)
        except Exception as e:
            warnings.warn(f"t_test_stat_p failed: {type(e).__name__}: {e}")
            return np.nan, np.nan

    def chi2_test_stat_p(series1, series2):
        """Chi-square test comparing a categorical variable across two groups.

        Takes two independent series (one per group) rather than a combined
        dataframe — avoids duplicate-index issues entirely. Builds the
        contingency table from per-category counts, so no pd.concat is needed.
        """
        try:
            # Drop NAs and convert to plain str (handles all nullable dtypes)
            s1 = series1.dropna().astype(str)
            s2 = series2.dropna().astype(str)
            if len(s1) == 0 or len(s2) == 0:
                return np.nan, np.nan
            # Union of categories present in either group
            categories = sorted(set(s1.unique()) | set(s2.unique()))
            counts1 = np.array([(s1 == cat).sum() for cat in categories])
            counts2 = np.array([(s2 == cat).sum() for cat in categories])
            contingency = np.array([counts1, counts2])
            # Fisher's exact only for 2x2 with small cell counts
            if contingency.shape in {(1, 2), (2, 2)}:
                if (contingency < 5).any():
                    _, p = stats.fisher_exact(
                        contingency.reshape(2, 2)
                        if contingency.shape != (2, 2)
                        else contingency
                    )
                    return np.nan, float(p)
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            return float(chi2), float(p)
        except Exception as e:
            warnings.warn(f"chi2_test_stat_p failed: {type(e).__name__}: {e}")
            return np.nan, np.nan

    def mann_whitney_stat_p(series1, series2):
        """Mann-Whitney U test, returns (U, p)."""
        try:
            s1 = pd.to_numeric(series1, errors="coerce").dropna().to_numpy(dtype=float)
            s2 = pd.to_numeric(series2, errors="coerce").dropna().to_numpy(dtype=float)
            if len(s1) < 2 or len(s2) < 2:
                return np.nan, np.nan
            u, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            return float(u), float(p)
        except Exception as e:
            warnings.warn(f"mann_whitney_stat_p failed: {type(e).__name__}: {e}")
            return np.nan, np.nan

    def fmt_t(stat, p):
        """Format t-statistic and p-value."""
        if pd.isna(p):
            return "NA"
        if pd.isna(stat):
            return f"p={p:.3f}"
        return f"t={stat:.2f}, p={p:.3f}"

    def fmt_chi2(stat, p):
        """Format chi-square (or Fisher) statistic and p-value."""
        if pd.isna(p):
            return "NA"
        if pd.isna(stat):
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
            sex_chi2, sex_p = chi2_test_stat_p(primary_df["sex"], df_subset["sex"])

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
                chi2, p = chi2_test_stat_p(
                    primary_df["sex"] == sex, df_subset["sex"] == sex
                )
                row["p-value"] = fmt_chi2(chi2, p)
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
            diag_chi2, diag_p = chi2_test_stat_p(
                primary_df[diagnosis_col], df_subset[diagnosis_col]
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
                chi2, p = chi2_test_stat_p(
                    primary_df[diagnosis_col] == diag,
                    df_subset[diagnosis_col] == diag,
                )
                row["p-value"] = fmt_chi2(chi2, p)
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
            smoking_chi2, smoking_p = chi2_test_stat_p(
                primary_df["smoker"], df_subset["smoker"]
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
                    chi2, p = chi2_test_stat_p(
                        primary_df["smoker"] == status,
                        df_subset["smoker"] == status,
                    )
                    row["p-value"] = fmt_chi2(chi2, p)
                rows.append(row)

    # === SECTION: MEDICATION ===
    rows.append(section_row("**Medication**"))

    # Antipsychotic use (any use: count > 0)
    if "antipsychotics_count" in primary_df.columns:
        any_ap_prs = primary_df["antipsychotics_count"] > 0
        row = {"Characteristic": "Antipsychotic use, n (%)"}
        row[primary_col] = format_count_pct(any_ap_prs, True, len(primary_df))
        if comparing:
            any_ap_lipid = df_subset["antipsychotics_count"] > 0
            row["Lipid Subset"] = format_count_pct(any_ap_lipid, True, len(df_subset))
            chi2, p = chi2_test_stat_p(any_ap_prs, any_ap_lipid)
            row["p-value"] = fmt_chi2(chi2, p)
        rows.append(row)

    # Mood stabilizer use (any use: count > 0)
    if "mood_stabilizers_count" in primary_df.columns:
        any_ms_prs = primary_df["mood_stabilizers_count"] > 0
        row = {"Characteristic": "Mood stabilizer use, n (%)"}
        row[primary_col] = format_count_pct(any_ms_prs, True, len(primary_df))
        if comparing:
            any_ms_lipid = df_subset["mood_stabilizers_count"] > 0
            row["Lipid Subset"] = format_count_pct(any_ms_lipid, True, len(df_subset))
            chi2, p = chi2_test_stat_p(any_ms_prs, any_ms_lipid)
            row["p-value"] = fmt_chi2(chi2, p)
        rows.append(row)

    # Antidepressant use (any use: count > 0)
    if "antidepressants_count" in primary_df.columns:
        any_ad_prs = primary_df["antidepressants_count"] > 0
        row = {"Characteristic": "Antidepressant use, n (%)"}
        row[primary_col] = format_count_pct(any_ad_prs, True, len(primary_df))
        if comparing:
            any_ad_lipid = df_subset["antidepressants_count"] > 0
            row["Lipid Subset"] = format_count_pct(any_ad_lipid, True, len(df_subset))
            chi2, p = chi2_test_stat_p(any_ad_prs, any_ad_lipid)
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
    *added_cov_dicts,
    covariate_names=None,
):
    """Takes the n_subset dicts from the regression analysis and puts them into a df.

    Args:
        standard_cov_dict (dict): dict with predictor (PRS or lipid) as key and
            n of individuals as value.
        *added_cov_dicts (dict): Optional dictionaries with predictor as key and
            n of individuals as value for additional covariate sets.
        covariate_names (list[str] | None): Optional names for additional
            covariate columns. If omitted, generic names are used.

    Returns:
        pd.DataFrame: Dataframe with predictors as rows and covariate sets as columns.
    """
    if covariate_names is None:
        covariate_names = [
            f"added_cov_{idx}" for idx in range(1, len(added_cov_dicts) + 1)
        ]
    elif len(covariate_names) != len(added_cov_dicts):
        raise ValueError(
            "Length mismatch: covariate_names must match number of added_cov_dicts."
        )

    dicts_dict = {"standard_cov": standard_cov_dict}
    dicts_dict.update(dict(zip(covariate_names, added_cov_dicts, strict=False)))

    n_per_analysis_df = pd.DataFrame(dicts_dict)

    return n_per_analysis_df
