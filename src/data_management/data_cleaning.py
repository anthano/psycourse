from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(".").resolve()
ROOT = THIS_DIR.parent.parent.resolve()
DATA_DIR = ROOT / "src" / "data"
BLD_DATA = ROOT / "bld" / "data"
BLD_DATA.mkdir(parents=True, exist_ok=True)


def clean_phenotypic_data(df):
    """Cleans the phenotypic data from '230614_v6.0_psycourse_wd.csv' by:

    - Selecting relevant columns based on criteria analogous to Dwyer et al. (2020).
    - Renaming columns for clarity.
    - Setting appropriate data types.

    Args:
        df (pd.DataFrame): The input dataframe containing raw phenotypic data.

    Returns:
        pd.DataFrame: The cleaned dataframe with selected and formatted columns.
    """
    clean_df = pd.DataFrame()
    clean_df["id"] = df["v1_id"].astype(str)
    clean_df["sex"] = df["v1_sex"].astype(pd.CategoricalDtype(categories=["F", "M"]))
    clean_df["age"] = df["v1_age"].astype(pd.Int8Dtype())
    clean_df["seas_birth"] = df["v1_seas_birth"].astype(
        pd.CategoricalDtype(categories=["Fall", "Spring", "Summer", "Winter"])
    )
    clean_df["age_m_birth"] = df["v1_age_m_birth"].astype(pd.Float32Dtype())
    clean_df["age_f_birth"] = df["v1_age_f_birth"].astype(pd.Float32Dtype())
    clean_df["marital_stat"] = (
        df["v1_marital_stat"]
        .replace(np.nan, pd.NA)
        .astype(
            pd.CategoricalDtype(
                categories=[
                    "Married",
                    "Single",
                    "Married_living_sep",
                    "Divorced",
                    "Widowed",
                ]
            )
        )
    )
    clean_df["partner"] = _map_yes_no(df["v1_partner"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["no_bio_children"] = (
        df["v1_no_bio_chld"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["no_step_children"] = (
        df["v1_stp_chld"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["no_adpt_children"] = (
        df["v1_no_adpt_chld"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["no_brothers"] = (
        df["v1_brothers"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["no_sisters"] = (
        df["v1_sisters"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["living_alone"] = _map_yes_no(df["v1_liv_aln"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["school"] = df["v1_school"].replace(np.nan, pd.NA)  # dtype?
    clean_df["ed_status"] = df["v1_ed_status"].replace(np.nan, pd.NA)  # dtype?
    clean_df["employment"] = _map_yes_no(df["v1_curr_paid_empl"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["disabilty_pension"] = _map_yes_no(df["v1_disabl_pens"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["work_absence"] = (
        df["v1_wrk_abs_pst_5_yrs"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["work_impairment"] = _map_yes_no(df["v1_cur_work_restr"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["current_psych_treatment"] = (
        df["v1_cur_psy_trm"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4"]))
    )
    clean_df["outpat_treatment"] = (
        df["v1_cur_psy_trm"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4"]))
    )
    clean_df["outpat_treatment_age"] = df["v1_age_1st_out_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["inpat_treatment"] = _map_yes_no(df["v1_daypat_inpat_trm"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["inpat_treatment_age"] = df["v1_age_1st_inpat_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["duration_illness"] = (
        df["v1_dur_illness"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["first_ep"] = _map_yes_no(df["v1_1st_ep"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["times_treated_inpatient_cont"] = (
        df["v1_tms_daypat_outpat_trm"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["times_treated_inpatient_ord"] = (
        df["v1_cat_daypat_outpat_trm"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4"]))
    )
    clean_df["adverse_events_curr_medication"] = _map_yes_no(df["v1_adv"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["med_change"] = _map_yes_no(df["v1_medchange"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["lithium"] = _map_yes_no(df["v1_lith"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["lithium_dur"] = (
        df["v1_lith_prd"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "-999"]))
    )
    clean_df["fam_hist"] = _map_yes_no(df["v1_fam_hist"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["height"] = df["v1_height"].astype(pd.Float32Dtype())
    clean_df["weight"] = df["v1_weight"].astype(pd.Float32Dtype())
    clean_df["bmi"] = df["v1_bmi"].astype(pd.Float32Dtype())
    ### add a bunch of other illness columns here

    clean_df["smoker"] = (
        df["v1_ever_smkd"]
        .replace({np.nan: pd.NA, "N": "never", "Y": "yes", "F": "former"})
        .astype(pd.CategoricalDtype(categories=["never", "yes", "former"]))
    )
    clean_df["no_cig"] = (
        df["v1_no_cig"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["alc_past_year"] = (
        df["v1_alc_pst12_mths"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "5", "6", "7"]))
    )
    clean_df["alc_5_drinks"] = (
        df["v1_alc_5orm"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(
            pd.CategoricalDtype(
                categories=["1", "2", "3", "4", "5", "6", "7", "8", "9", "-999"]
            )
        )
    )
    clean_df["alc_dependence"] = _map_yes_no(df["v1_lftm_alc_dep"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["illicit_drugs"] = _map_yes_no(df["v1_evr_ill_drg"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["age_at_mde"] = (
        df["v1_scid_age_MDE"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["age_at_mania"] = (
        df["v1_scid_age_mania"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["no_of_mania"] = (
        df["v1_scid_no_mania"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["age_at_hypomania"] = (
        df["v1_scid_age_hypomania"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["no_of_hypomania"] = (
        df["v1_scid_no_hypomania"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )

    return clean_df


def _map_yes_no(sr):
    """Maps the 'currently has a partner' column to "yes", "no", "pd.NA" """

    mapping_partner = {"-999": "-999", "N": "no", "Y": "yes", np.nan: pd.NA}

    return sr.map(mapping_partner)


def _map_pdNA(sr):
    """Maps the 'nan' values to pd.NA"""

    mapping_pdna = {np.nan: pd.NA}

    return sr.map(mapping_pdna)


if __name__ == "__main__":
    pass
