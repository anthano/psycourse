import pandas as pd


def concat_full_predicted_probability_df(old_df_probabilities, new_df_probabilities):
    """
    Concatenate the predicted probabilities DataFrames for old and new psycourse data.

    Args:
    - old_df_probabilities:
    DataFrame with predicted probabilities for the original psycoures data.
    - new_df_probabilities:
    DataFrame with predicted probabilities for new psycourse data.

    Returns:
    - concatenated_df: Concatenated DataFrame with all predicted probabilities.
    """
    concatenated_df = pd.concat([old_df_probabilities, new_df_probabilities], axis=0)

    return concatenated_df
