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
        "LPE",
        "PC",
        "PC_O",
        "PC_P",
        "PE",
        "PE_P",
        "TAG",
        "dCer",
        "dSM",
        "CAR",
        "CE",
        "DAG",
        "FA",
        "LPC",
        "LPC_O",
        "LPC_P",
    ]
    data_with_lipids = data[~data[lipid_features].isna().all(axis=1)]
    return data_with_lipids
