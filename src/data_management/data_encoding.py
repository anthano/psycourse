from pathlib import Path

import pandas as pd

THIS_DIR = Path(".").resolve()
ROOT = THIS_DIR.parent.parent.resolve()
DATA_DIR = ROOT / "src" / "data"
BLD_DATA = ROOT / "bld" / "data"
BLD_DATA.mkdir(parents=True, exist_ok=True)


def encode_data(df):
    """Takes the cleaned phenotypic dataframes and encodes them for future model
    statistics.

    Args:
        df (pd.DataFrame): The cleaned phenotypic dataframe.

    Returns:
        pd.DataFrame: The encoded dataframe.

    """
    encoded_df = df.copy()
    encoded_df["sex"] = encoded_df["sex"].map({"F": 0, "M": 1})

    variables_to_encode = ["seas_birth", "marital_stat"]
    for variable in variables_to_encode:
        encoded_df = pd.get_dummies(
            encoded_df, columns=[variable], dtype=pd.Int8Dtype()
        )

    # variables_to_boolean = ["partner"]

    return encoded_df


def _map_yes_no(sr):
    """Maps yes to 1 and no to 0."""
    return sr.map({"yes": 1, "no": 0}).astype(pd.Int8Dtype())
