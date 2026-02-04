import pandas as pd


def get_demographics_table(df):
    """Get table with participant characteristics.
    Args: df (pd.DataFrame): Descriptive df of participants with all variables.

    Returns: df(pd.DataFrame): Condensed table with descriptives, mean, sd for subset
    of demographic variables."""

    rows = []

    rows.append(("N", f"{len(df)}"))

    if "age" in df:
        rows.append(
            ("Age, mean (SD)", f"{df['age'].mean():.1f} ({df['age'].std():.1f})")
        )

    if "sex" in df:
        counts = df["sex"].value_counts(dropna=False)
        for k, v in counts.items():
            rows.append((f"Sex: {k}", f"{v} ({v/len(df)*100:.1f}%)"))

    if "bmi" in df:
        rows.append(
            ("BMI, mean (SD)", f"{df['bmi'].mean():.1f} ({df['bmi'].std():.1f})")
        )

    if "smoker" in df:
        counts = df["smoker"].value_counts(dropna=False)
        for k, v in counts.items():
            rows.append((f"Smoking: {k}", f"{v} ({v/len(df)*100:.1f}%)"))

    return pd.DataFrame(rows, columns=[" ", " "])


def create_lipid_table(cleaned_lipid_class_data):
    """Already exists as pickle.
    Load it, fix index, export to csv.
    """
    lipid_df = cleaned_lipid_class_data.copy()
    # lipid_df = pd.read_pickle(BLD_DATA / "cleaned_lipid_class_data.pkl")
    lipid_df["lipid"] = lipid_df.index

    return lipid_df


def get_participants_per_analysis(
    standard_cov_dict,
    added_cov_1_dict,
    added_cov_2_dict,
    cov_1_name="added_cov_1",
    cov_2_name="added_cov_2",
):
    """Takes the n_subset dicts from the regression analysis and puts them into a df.

    Args:
        standard_cov_dict (dict): dict with predictor (PRS or lipid) as key and
            n of individuals as value.
        added_cov_1_dict (dict): dict with predictor as key and n of individuals
            as value (first additional covariate set).
        added_cov_2_dict (dict): dict with predictor as key and n of individuals
            as value (second additional covariate set).
        cov_1_name (str): Name for first additional covariate set column.
        cov_2_name (str): Name for second additional covariate set column.

    Returns:
        pd.DataFrame: Dataframe with predictors as rows and covariate sets as columns.
    """

    dicts_dict = {
        "standard_cov": standard_cov_dict,
        cov_1_name: added_cov_1_dict,
        cov_2_name: added_cov_2_dict,
    }

    n_per_analysis_df = pd.DataFrame(dicts_dict)

    return n_per_analysis_df
