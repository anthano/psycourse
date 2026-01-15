import pandas as pd


def prep_data_for_integrated_analysis(multimodal_lipid_subset_df):
    """
    Prepare three separate dataframes for each modality for the integrated analysis.
    Args:
        multimodal_lipid_subset_df(pd.DataFrame): multimodal dataframe w/ lipid subset.
    Returns:
        lipid_data(pd.DataFrame): dataframe for lipid modality.
        prs_data(pd.DataFrame): dataframe for PRS modality.
        outcome_data(pd.DataFrame): dataframe for subgroup labels.
    """

    lipid_df = _prep_lipid_data(multimodal_lipid_subset_df)
    prs_df = _prep_prs_data(multimodal_lipid_subset_df)
    outcome_df = _prep_outcome_data(multimodal_lipid_subset_df)

    return lipid_df, prs_df, outcome_df


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _prep_lipid_data(multimodal_lipid_subset_df):
    lipid_columns = [
        col for col in multimodal_lipid_subset_df.columns if "gpeak" in col
    ]
    lipid_df = pd.DataFrame(index=multimodal_lipid_subset_df.index)
    for col in lipid_columns:
        lipid_df = lipid_df.join(multimodal_lipid_subset_df[col])

    return lipid_df


def _prep_prs_data(multimodal_lipid_subset_df):
    prs_columns = [col for col in multimodal_lipid_subset_df.columns if "PRS" in col]
    prs_df = pd.DataFrame(index=multimodal_lipid_subset_df.index)
    for col in prs_columns:
        prs_df = prs_df.join(multimodal_lipid_subset_df[col])

    return prs_df


def _prep_outcome_data(multimodal_lipid_subset_df):
    outcome_df = multimodal_lipid_subset_df[["true_label"]].copy()
    return outcome_df
