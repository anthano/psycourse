import pandas as pd


def merge_multimodal_complete_df(
    lipid_data, phenotypic_data, cluster_probabilities_full, prs_data
):
    """
    Merges lipid data, phenotypic data, prs and cluster probabilites into a single
    dataframe for data analysis.
    Args:
        lipid_data (pd.DataFrame): Dataframe containing lipid data.
        phenotypic_data (pd.DataFrame): Dataframe containing phenotypic data.
        cluster_probabilites (pd.DataFrame): Dataframe containing cluster probabilities,
        incl. the newly obtained (full).
        prs (pd.DataFrame): Dataframe containing polygenic risk scores.
    Returns:
        pd.DataFrame: Merged dataframe containing lipid data, phenotypic data, prs
        and cluster probabilites.
    """

    multimodal_df = pd.DataFrame(index=phenotypic_data.index)
    multimodal_df["sex"] = phenotypic_data["sex"]
    multimodal_df["age"] = phenotypic_data["age"]
    multimodal_df["bmi"] = phenotypic_data["bmi"]
    multimodal_df["gsa_id"] = phenotypic_data["gsa_id"]
    multimodal_df["diagnosis"] = phenotypic_data["diagnosis"]

    multimodal_df = multimodal_df.join(cluster_probabilities_full, how="left")

    lipids_df_to_join = lipid_data.drop(columns=["sex", "age"])
    multimodal_df = multimodal_df.join(lipids_df_to_join, how="left")

    multimodal_df = multimodal_df.reset_index()
    multimodal_df = multimodal_df.set_index("gsa_id")
    multimodal_df = multimodal_df.join(prs_data, how="left")
    multimodal_df = multimodal_df.set_index("id")

    return multimodal_df
