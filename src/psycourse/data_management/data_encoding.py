import pandas as pd


def encode_and_prune_data(df):
    """Takes the cleaned phenotypic dataframes and encodes them for future model
    statistics. Additionally, removes the columns that have (1) more than 25% missing or
    (2) limited variance (> 95% of the values are the same).

    Args:
        df (pd.DataFrame): The cleaned phenotypic dataframe.

    Returns:
        pd.DataFrame: The encoded dataframe.

    """
    encoded_df = df.copy()

    # ==================================================================================
    # Create binary variables
    # ==================================================================================
    encoded_df["sex"] = encoded_df["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    encoded_df["ever_mde"] = encoded_df["age_at_mde"].gt(0).astype(pd.Int8Dtype())
    encoded_df["ever_mania"] = encoded_df["age_at_mania"].gt(0).astype(pd.Int8Dtype())
    encoded_df["ever_hypomania"] = (
        encoded_df["age_at_hypomania"].gt(0).astype(pd.Int8Dtype())
    )

    # ==================================================================================
    # Change the -999 values to NaN
    # ==================================================================================

    encoded_df["age_at_mde"] = (
        encoded_df["age_at_mde"].replace(-999, pd.NA).astype(pd.Int8Dtype())
    )
    encoded_df["age_at_mania"] = (
        encoded_df["age_at_mania"].replace(-999, pd.NA).astype(pd.Int8Dtype())
    )
    encoded_df["ever_hypomania"] = (
        encoded_df["ever_hypomania"].replace(-999, pd.NA).astype(pd.Int8Dtype())
    )

    # ----------------------------------------------------------------------------------
    # Perform one-hot encoding on categorical variables with more than 2 categories
    # ----------------------------------------------------------------------------------
    variables_to_encode = ["seas_birth", "marital_stat", "smoker", "language_skill"]
    encoded_df = pd.get_dummies(
        encoded_df, columns=variables_to_encode, dtype=pd.Int8Dtype()
    )

    # ----------------------------------------------------------------------------------
    # Convert dichotomous variables to 0-1
    # ----------------------------------------------------------------------------------

    dichotomous_variables = [
        "partner",
        "living_alone",
        "employment",
        "disability_pension",
        "supported_employment",
        "work_impairment",
        "first_ep",
        "ever_delus",
        "ever_halluc",
        "ever_psyc",
        "ever_suic_ide",
        "inpat_treatment",
        "adverse_events_curr_medication",
        "med_change",
        "lithium",
        "fam_hist",
        "alc_dependence",
        "illicit_drugs",
        "cholesterol",
        "hypertension",
        "angina_pectoris",
        "heartattack",
        "hypothyroid",
        "hyperthyroid",
        "asthma",
        "copd",
        "allergies",
        "osteoporosis",
        "diabetes",
        "stomach_ulc",
        "liv_cir_inf",
        "neuroderm",
        "psoriasis",
        "kidney_fail",
        "stone",
        "epilepsy",
        "migraine",
        "parkinson",
        "tbi",
        "stroke",
        "eyear",
        "inf",
        "cancer",
        "autoimm",
        "rel_christianity",
        "rel_islam",
        "rel_other",
        "beh",
    ]

    for variable in dichotomous_variables:
        encoded_df[variable] = encoded_df[variable].cat.remove_unused_categories()
        encoded_df[variable] = _map_yes_no(encoded_df[variable])

    # ----------------------------------------------------------------------------------
    # Create summary variables for disease categories
    # ----------------------------------------------------------------------------------

    cd_cols = ["cholesterol", "hypertension", "angina_pectoris", "heartattack"]
    metabolic_cols = ["diabetes"]
    thyroid_cols = ["hypothyroid", "hyperthyroid"]
    rheumatological_cols = ["osteoporosis"]
    lung_cols = ["asthma", "copd"]
    allergies_cols = ["allergies"]
    gastro_cols = ["stomach_ulc"]
    liver_cols = ["liv_cir_inf"]
    skin_cols = ["neuroderm", "psoriasis"]
    kidney_cols = ["kidney_fail", "stone"]
    neuro_cols = ["epilepsy", "migraine", "parkinson", "tbi", "stroke"]
    eyear_cols = ["eyear"]
    inf_cols = ["inf"]
    cancer_cols = ["cancer"]
    other_cols = ["autoimm", "beh"]

    encoded_df["cardiovascular_disease"] = (encoded_df[cd_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["metabolic_disorder"] = (
        encoded_df[metabolic_cols].sum(axis=1) > 0
    ).astype(pd.Int8Dtype())
    encoded_df["thyroid_disorder"] = (encoded_df[thyroid_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["rheumatological_disorder"] = (
        encoded_df[rheumatological_cols].sum(axis=1) > 0
    ).astype(pd.Int8Dtype())
    encoded_df["lung_disorder"] = (encoded_df[lung_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["allergies"] = (encoded_df[allergies_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["gastrointestinal_disorder"] = (
        encoded_df[gastro_cols].sum(axis=1) > 0
    ).astype(pd.Int8Dtype())
    encoded_df["liver_disease"] = (encoded_df[liver_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["skin_disorder"] = (encoded_df[skin_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["kidney_disorder"] = (encoded_df[kidney_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["neuro_disorder"] = (encoded_df[neuro_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["eyear_disease"] = (encoded_df[eyear_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["inf_disease"] = (encoded_df[inf_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["cancer"] = (encoded_df[cancer_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )
    encoded_df["other"] = (encoded_df[other_cols].sum(axis=1) > 0).astype(
        pd.Int8Dtype()
    )

    # --------------------------------------------------------------------------------
    # Drop the individual disease columns + low variance/high missingness columns
    # --------------------------------------------------------------------------------
    cols_to_drop = [
        *cd_cols,
        *metabolic_cols,
        *thyroid_cols,
        *rheumatological_cols,
        *lung_cols,
        *allergies_cols,
        *gastro_cols,
        *liver_cols,
        *skin_cols,
        *kidney_cols,
        *neuro_cols,
        *eyear_cols,
        *inf_cols,
        *cancer_cols,
        *other_cols,
    ]

    low_var_high_na_cols = _identify_low_variance_high_na_cols(encoded_df)

    all_cols_to_drop = set(cols_to_drop) | set(low_var_high_na_cols)
    encoded_df = encoded_df.drop(columns=list(all_cols_to_drop))

    return encoded_df


def _map_yes_no(sr):
    """Maps yes to 1 and no to 0."""
    return sr.map({"yes": 1, "no": 0}).astype(pd.Int8Dtype())


def _identify_low_variance_high_na_cols(df):
    """
    Identifies columns that have low variance (95% of non-missing values are the same)
    or high missingness (>25% missing values), and returns a list of those columns.
    """
    low_var_cols = []
    high_na_cols = []

    for col in df.columns:
        if df[col].isna().mean() > 0.25:
            high_na_cols.append(col)

    if not df[col].dropna().empty:
        if df[col].dropna().value_counts(normalize=True).max() >= 0.95:
            low_var_cols.append(col)

    # Return the union of both lists without duplicates
    return list(set(low_var_cols + high_na_cols))
