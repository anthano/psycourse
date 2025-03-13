import pandas as pd


def concatenate_features_and_targets(phenotypic_df, target_df):
    """Concatenate the features and targets into a single dataframe." """

    return pd.concat([phenotypic_df, target_df], axis=1, join="inner")
