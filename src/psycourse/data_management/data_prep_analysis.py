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

    multimodal_df = multimodal_df.join(cluster_probabilities_full, how="left")

    lipids_df_to_join = lipid_data.drop(columns=["sex", "age"])
    multimodal_df = multimodal_df.join(lipids_df_to_join, how="left")

    multimodal_df = multimodal_df.reset_index()
    multimodal_df = multimodal_df.set_index("gsa_id")
    multimodal_df = multimodal_df.join(prs_data, how="left")

    for pc in range(1, 11):
        multimodal_df[f"pc{pc}"] = pc_components[f"PC{pc}"]

    multimodal_df = multimodal_df.set_index("id")

    class_means_df = compute_lipid_class_means(lipid_data, lipid_class)
    multimodal_df = multimodal_df.join(class_means_df, how="left")
    multimodal_df["gsa_id"] = phenotypic_data["gsa_id"]

    return multimodal_df


##################### Helper Function  #####################


def compute_lipid_class_means(intensity_df, lipid_class):
    lipid_dict = lipid_class.groupby("class").apply(lambda g: list(g.index)).to_dict()
    class_means = {}

    for class_name, species_list in lipid_dict.items():
        # Only include species that exist in the intensity_df
        valid_species = [
            lipid for lipid in species_list if lipid in intensity_df.columns
        ]
        if not valid_species:
            continue
        # Compute mean across those species for each individual
        class_means[f"{class_name}_mean"] = intensity_df[valid_species].mean(axis=1)

    class_means_df = pd.DataFrame(class_means, index=intensity_df.index)

    return class_means_df
