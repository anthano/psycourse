import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Ordinal codes used when encoding categorical covariates for KNN imputation.
# Sex: CategoricalDtype(["F", "M"]) → 0/1 and back.
_SEX_ENCODE = {"F": 0, "M": 1}
_SEX_DECODE = {0: "F", 1: "M"}
# Smoker: CategoricalDtype(["never", "former", "yes"]) → 0/1/2 ordinal and back.
_SMOKER_ENCODE = {"never": 0, "former": 1, "yes": 2}
_SMOKER_DECODE = {0: "never", 1: "former", 2: "yes"}


def merge_multimodal_complete_df(
    lipid_data,
    lipid_class,
    phenotypic_data,
    cluster_probabilities_full,
    prs_data,
    pc_components,
):
    """
    Merges lipid data, phenotypic data, prs and cluster probabilites into a single
    dataframe for data analysis.
    Args:
        lipid_data (pd.DataFrame): Dataframe containing lipid data.
        lipid_class (pd.DataFrame): Dataframe containing lipid class information.
        phenotypic_data (pd.DataFrame): Dataframe containing phenotypic data.
        cluster_probabilites (pd.DataFrame): Dataframe containing cluster probabilities,
        incl. the newly obtained (full).
        prs (pd.DataFrame): Dataframe containing polygenic risk scores.
        pc_components (pd.DataFrame): Dataframe containing
        ancestry principal component scores.
    Returns:
        pd.DataFrame: Merged dataframe containing lipid data, phenotypic data, prs
        and cluster probabilites.
    """

    multimodal_df = pd.DataFrame(index=phenotypic_data.index)
    multimodal_df["sex"] = phenotypic_data["sex"]
    multimodal_df["age"] = phenotypic_data["age"]
    multimodal_df["bmi"] = phenotypic_data["bmi"]
    multimodal_df["smoker"] = phenotypic_data["smoker"]
    multimodal_df["duration_illness"] = phenotypic_data["duration_illness"]
    multimodal_df["gsa_id"] = phenotypic_data["gsa_id"]
    multimodal_df["diagnosis"] = phenotypic_data["diagnosis"]
    multimodal_df["diagnosis_sum"] = phenotypic_data["diagnosis_sum"]
    multimodal_df["antidepressants_count"] = phenotypic_data["antidepressants_count"]
    multimodal_df["antipsychotics_count"] = phenotypic_data["antipsychotics_count"]
    multimodal_df["mood_stabilizers_count"] = phenotypic_data["mood_stabilizers_count"]
    multimodal_df["tranquilizers_count"] = phenotypic_data["tranquilizers_count"]
    multimodal_df["other_psy_med_count"] = phenotypic_data["other_psy_med_count"]
    multimodal_df["panss_sum_pos"] = phenotypic_data["panss_sum_pos"]
    multimodal_df["panss_sum_neg"] = phenotypic_data["panss_sum_neg"]
    multimodal_df["panss_sum_gen"] = phenotypic_data["panss_sum_gen"]
    multimodal_df["panss_total_score"] = phenotypic_data["panss_total_score"]

    multimodal_df = multimodal_df.join(cluster_probabilities_full, how="left")

    lipids_df_to_join = lipid_data.drop(columns=["sex", "age"])
    multimodal_df = multimodal_df.join(lipids_df_to_join, how="left")

    multimodal_df = multimodal_df.reset_index()
    multimodal_df = multimodal_df.set_index("gsa_id")
    multimodal_df = multimodal_df.join(prs_data, how="left")

    for pc in range(1, 11):
        multimodal_df[f"pc{pc}"] = pc_components[f"PC{pc}"]

    multimodal_df = multimodal_df.set_index("id")

    class_means_df = _lipid_class_scores(lipid_data, lipid_class)
    multimodal_df = multimodal_df.join(class_means_df, how="left")
    multimodal_df["gsa_id"] = phenotypic_data["gsa_id"]

    multimodal_df_imputed = _impute_covariates(multimodal_df)

    return multimodal_df_imputed


##################### Helper Function  #####################


def _lipid_class_scores(
    lipid_df: pd.DataFrame,
    lipid_class_scores_df: pd.DataFrame,
    class_col: str = "class",
    min_frac_present: float = 0.7,
    ddof: int = 0,
) -> pd.DataFrame:
    """
    lipid_df: rows=samples, cols=lipid features (raw lipid names)
    lipid_class_df: index = raw lipid names (must match lipid_df columns),
                    column `class_col` contains the class label per lipid.
    Returns: DataFrame of per-sample class scores (mean of within-lipid z-scores).
    """

    # keep only lipids that exist in lipid_df
    if class_col not in lipid_class_scores_df.columns:
        raise KeyError(f"Column '{class_col}' not found in lipid_class_scores_df")
    m = lipid_class_scores_df[[class_col]].copy()
    m = m.loc[m.index.intersection(lipid_df.columns)]
    m = m.dropna(subset=[class_col])

    lipid_cols = m.index.tolist()

    # z-score each lipid across samples (ignore NaNs)
    X = lipid_df[lipid_cols].copy()
    mu = X.mean(axis=0, skipna=True)
    sd = X.std(axis=0, ddof=ddof, skipna=True).replace(0, np.nan)
    Xz = (X - mu) / sd

    lipid_class_scores_df = pd.DataFrame(index=lipid_df.index)

    for cls, idx in m.groupby(class_col).groups.items():
        cols = list(idx)  # lipid names for this class

        present_frac = Xz[cols].notna().mean(axis=1)
        score = Xz[cols].mean(axis=1, skipna=True)

        score[present_frac < min_frac_present] = np.nan
        lipid_class_scores_df[f"{cls}"] = score

    return lipid_class_scores_df


def _impute_covariates(
    df: pd.DataFrame,
    covariate_cols: list[str] | None = None,
    n_neighbors: int = 7,
    weights: str = "uniform",
) -> pd.DataFrame:
    """
    Imputes missing values in covariate columns using KNN imputation.
    Lipid species, lipid class scores, and PRS columns are explicitly
    excluded and left unchanged, as missingness in these modalities
    is not assumed to be missing at random.

    Categorical covariates are handled as follows before imputation:
      - sex (CategoricalDtype ["F", "M"]): encoded as 0/1, imputed, rounded,
        and decoded back to CategoricalDtype(["F", "M"]).
      - smoker (CategoricalDtype ["never", "former", "yes"]): encoded
        ordinally as 0/1/2, imputed, rounded to the nearest integer,
        and decoded back to CategoricalDtype(["never", "former", "yes"]).
        This preserves the three-level structure so that downstream
        pd.get_dummies produces two dummy columns rather than treating
        smoker as a continuous predictor.

    Args:
        df: Merged multimodal DataFrame as returned by
            merge_multimodal_complete_df.
        covariate_cols: Columns to impute. Defaults to the standard
            covariate set used in primary models (age, sex, bmi,
            duration_illness, smoker).
        n_neighbors: Number of neighbours for KNN imputation (default 7).
        weights: Weight function for KNN — 'uniform' weights all
            neighbours equally; 'distance' weights by inverse distance.

    Returns:
        pd.DataFrame: Copy of df with covariate missingness imputed.
        Original df is not modified.
    """
    if covariate_cols is None:
        covariate_cols = ["age", "sex", "bmi", "duration_illness", "smoker"]

    missing_cols = [c for c in covariate_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Covariate columns not found in DataFrame: {missing_cols}")

    df_out = df.copy()

    # Encode sex to 0/1. sex is stored as CategoricalDtype(["F", "M"]),
    # so dtype == object is False — check for CategoricalDtype explicitly.
    sex_was_categorical = False
    if "sex" in df_out.columns and (
        hasattr(df_out["sex"].dtype, "categories") or df_out["sex"].dtype == object
    ):
        df_out["sex"] = df_out["sex"].map(_SEX_ENCODE)
        sex_was_categorical = True

    # Encode smoker to 0/1/2. smoker is stored as
    # CategoricalDtype(["never", "former", "yes"]), so pd.to_numeric would
    # coerce every valid value to NaN — map explicitly before numeric conversion.
    smoker_was_categorical = False
    if "smoker" in df_out.columns and (
        hasattr(df_out["smoker"].dtype, "categories")
        or df_out["smoker"].dtype == object
    ):
        df_out["smoker"] = df_out["smoker"].map(_SMOKER_ENCODE)
        smoker_was_categorical = True

    # Convert all covariate columns to plain float64 (handles nullable Int8/Float32
    # dtypes that KNNImputer cannot accept directly).
    for col in covariate_cols:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # KNN imputation. Wrap the numpy output in a DataFrame that carries the
    # original index so that the assignment back to df_out is index-safe.
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed = imputer.fit_transform(df_out[covariate_cols])
    df_out[covariate_cols] = pd.DataFrame(
        imputed, index=df_out.index, columns=covariate_cols
    )

    # Restore sex: round to nearest int, clip to valid range {0, 1}, decode
    # back to CategoricalDtype so downstream pd.get_dummies works correctly.
    if sex_was_categorical and "sex" in df_out.columns:
        df_out["sex"] = (
            df_out["sex"]
            .round()
            .clip(0, 1)
            .astype(int)
            .map(_SEX_DECODE)
            .astype(pd.CategoricalDtype(categories=["F", "M"]))
        )

    # Restore smoker: round to nearest int, clip to valid range {0, 1, 2},
    # decode back to CategoricalDtype so the three-level structure is preserved.
    if smoker_was_categorical and "smoker" in df_out.columns:
        df_out["smoker"] = (
            df_out["smoker"]
            .round()
            .clip(0, 2)
            .astype(int)
            .map(_SMOKER_DECODE)
            .astype(pd.CategoricalDtype(categories=["never", "former", "yes"]))
        )

    return df_out
