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

    # ==================================================================================
    # Create binary variables
    # ==================================================================================
    encoded_df["sex"] = encoded_df["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())

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

    encoded_df["ever_mde"] = encoded_df["age_at_mde"].gt(0).astype(pd.Int8Dtype())
    encoded_df["ever_mania"] = encoded_df["age_at_mania"].gt(0).astype(pd.Int8Dtype())
    encoded_df["ever_hypomania"] = (
        encoded_df["age_at_hypomania"].gt(0).astype(pd.Int8Dtype())
    )

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

    encoded_df = encoded_df.drop(columns=cols_to_drop)

    return encoded_df


def _map_yes_no(sr):
    """Maps yes to 1 and no to 0."""
    return sr.map({"yes": 1, "no": 0}).astype(pd.Int8Dtype())


if __name__ == "__main__":
    pass
