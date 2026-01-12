def subset_multimodal_data_for_lipids(multimodal_df):
    """
    Subset the multimodal dataframe to only contain participants with lipid data.

    Args:
        multimodal_df (pd.DataFrame): The input multimodal dataframe.

    Returns:
        pd.DataFrame: A dataframe containing only the participants with lipid data.
    """
    data = multimodal_df.copy()
    lipid_features = [
        "class_LPE",
        "class_PC",
        "class_PC_O",
        "class_PC_P",
        "class_PE",
        "class_PE_P",
        "class_TAG",
        "class_dCer",
        "class_dSM",
        "class_CAR",
        "class_CE",
        "class_DAG",
        "class_FA",
        "class_LPC",
        "class_LPC_O",
        "class_LPC_P",
    ]
    data_with_lipids = data[~data[lipid_features].isna().all(axis=1)]
    return data_with_lipids
