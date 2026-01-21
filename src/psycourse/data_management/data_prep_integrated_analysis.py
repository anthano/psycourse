import pandas as pd
from sklearn.model_selection import train_test_split


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

    multimodal_lipid_subset_df = multimodal_lipid_subset_df.copy()
    y = (multimodal_lipid_subset_df["true_label"] == 5).astype(
        int
    )  # 1 = label 5, 0 = rest

    X_train, X_test, y_train, y_test = train_test_split(
        multimodal_lipid_subset_df,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
        shuffle=True,
    )

    lipid_df_train = _prep_lipid_data(X_train)
    lipid_class_df_train = _prep_lipid_class_data(X_train)
    prs_df_train = _prep_prs_data(X_train)
    lipid_df_test = _prep_lipid_data(X_test)
    lipid_class_df_test = _prep_lipid_class_data(X_test)
    prs_df_test = _prep_prs_data(X_test)
    outcome_df_test = y_test
    outcome_df_train = y_train

    return (
        lipid_df_test,
        lipid_class_df_test,
        prs_df_test,
        outcome_df_test,
        lipid_df_train,
        lipid_class_df_train,
        prs_df_train,
        outcome_df_train,
    )


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


def _prep_lipid_class_data(multimodal_lipid_subset_df):
    lipid_columns = [
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

    lipid_class_df = pd.DataFrame(index=multimodal_lipid_subset_df.index)
    for col in lipid_columns:
        lipid_class_df = lipid_class_df.join(multimodal_lipid_subset_df[col])

    return lipid_class_df


def _prep_prs_data(multimodal_lipid_subset_df):
    prs_columns = [col for col in multimodal_lipid_subset_df.columns if "PRS" in col]
    prs_df = pd.DataFrame(index=multimodal_lipid_subset_df.index)
    for col in prs_columns:
        prs_df = prs_df.join(multimodal_lipid_subset_df[col])

    return prs_df


def _prep_outcome_data(multimodal_lipid_subset_df):
    outcome_df = multimodal_lipid_subset_df[["true_label"]].copy()
    return outcome_df
