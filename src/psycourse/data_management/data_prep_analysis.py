import numpy as np
import pandas as pd


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

    return multimodal_df


##################### Helper Function  #####################


def _lipid_class_scores(
    lipid_df: pd.DataFrame,
    lipid_class_scores_df: pd.DataFrame,
    class_col: str = "class",
    prefix: str = "class_",
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
        lipid_class_scores_df[f"{prefix}{cls}"] = score

    return lipid_class_scores_df
