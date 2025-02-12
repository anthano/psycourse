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

    clean_df["ever_delus"] = _map_yes_no(df["v1_scid_ever_delus"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["ever_halluc"] = _map_yes_no(df["v1_scid_ever_halls"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["ever_psyc"] = _map_yes_no(df["v1_scid_ever_psyc"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["ever_suic_ide"] = _map_yes_no(df["v1_scid_evr_suic_ide"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )
    clean_df["severity_suic_ide"] = (
        df["v1_scid_suic_ide"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "-999"]))
    )
    clean_df["suic_methods"] = (
        df["v1_scid_suic_thght_mth"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "-999"]))
    )
    clean_df["suic_note"] = (
        df["v1_scid_suic_note_thgts"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "-999"]))
    )
    clean_df["suic_attempt"] = (
        df["v1_suic_attmpt"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "-999"]))
    )
    clean_df["no_suic_attempt"] = (
        df["v1_scid_no_suic_attmpt"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "5", "6", "-999"]))
    )
    clean_df["prep_suic_attempt_ord"] = (
        df["v1_prep_suic_attp_ord"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "-999"]))
    )
    clean_df["suic_attempt_note"] = (
        df["v1_suic_note_attmpt"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "-999"]))
    )
    clean_df["panss_total_score"] = (
        df["v1_panss_sum_tot"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )

    idsc_dtype = pd.CategoricalDtype(categories=["0", "1", "2", "3"], ordered=True)

    for i in range(1, 10):
        clean_df[f"idsc_{i}"] = (
            df[f"v1_idsc_itm{i}"]
            .replace(np.nan, pd.NA)
            .astype("Int64")
            .astype("string")
            .astype(idsc_dtype)
        )

    clean_df["idsc_9a"] = (
        df["v1_idsc_itm9a"]
        .replace(np.nan, pd.NA)
        .astype(pd.CategoricalDtype(categories=["-999", "A", "M", "N"]))
    )
    clean_df["idsc_9b"] = _map_yes_no(df["v1_idsc_itm9b"]).astype(
        pd.CategoricalDtype(categories=["yes", "no", "-999"])
    )

    for i in range(10, 31):
        clean_df[f"idsc_{i}"] = (
            df[f"v1_idsc_itm{i}"]
            .replace(np.nan, pd.NA)
            .astype("Int64")
            .astype("string")
            .astype(idsc_dtype)
        )

    clean_df["idsc_total"] = (
        df["v1_idsc_sum"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )

    ymrs_dtype_single = pd.CategoricalDtype(
        categories=["0", "1", "2", "3", "4"], ordered=True
    )
    ymrs_dtype_double = pd.CategoricalDtype(
        categories=["0", "2", "4", "6", "8"], ordered=True
    )
    ymrs_single_items = [1, 2, 3, 4, 7, 10, 11]
    ymrs_double_items = [5, 6, 8, 9]
    for i in ymrs_single_items:
        clean_df[f"ymrs_{i}"] = (
            df[f"v1_ymrs_itm{i}"]
            .replace(np.nan, pd.NA)
            .astype("Int64")
            .astype("string")
            .astype(ymrs_dtype_single)
        )

    for i in ymrs_double_items:
        clean_df[f"ymrs_{i}"] = (
            df[f"v1_ymrs_itm{i}"]
            .replace(np.nan, pd.NA)
            .astype("Int64")
            .astype("string")
            .astype(ymrs_dtype_double)
        )

    clean_df["ymrs_total"] = (
        df["v1_ymrs_sum"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["cgi"] = (
        df["v1_cgi_s"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(
            pd.CategoricalDtype(
                categories=["1", "2", "3", "4", "5", "6", "7", "-999"], ordered=True
            )
        )
    )

    clean_df["gaf"] = df["v1_gaf"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    clean_df["language_skill"] = df["v1_nrpsy_lng"].replace(np.nan, pd.NA)
    clean_df["tmt_a_time"] = (
        df["v1_nrpsy_tmt_A_rt"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["tmt_a_err"] = (
        df["v1_nrpsy_tmt_A_err"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["tmt_b_time"] = (
        df["v1_nrpsy_tmt_B_rt"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["tmt_b_err"] = (
        df["v1_nrpsy_tmt_B_err"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["dgt_sp_fwd"] = (
        df["v1_nrpsy_dgt_sp_frw"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["dgt_sp_bck"] = (
        df["v1_nrpsy_dgt_sp_bck"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["dst"] = (
        df["v1_nrpsy_dg_sym"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["mwtb"] = (
        df["v1_nrpsy_mwtb"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["rel_christianity"] = _map_yes_no(df["v1_rel_chr"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["rel_islam"] = _map_yes_no(df["v1_rel_isl"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["rel_other"] = _map_yes_no(df["v1_rel_oth"]).astype(
        pd.CategoricalDtype(categories=["yes", "no"])
    )
    clean_df["rel_act"] = (
        df["v1_rel_act"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "5"]))
    )
    clean_df["med_compliance_week"] = (
        df["v1_med_pst_wk"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "5", "6", "-999"]))
    )
    clean_df["med_compliance_6_months"] = (
        df["v1_med_pst_sx_mths"]
        .replace(np.nan, pd.NA)
        .astype("Int64")
        .astype("string")
        .astype(pd.CategoricalDtype(categories=["1", "2", "3", "4", "5", "6", "-999"]))
    )

    whoqol_cat = pd.CategoricalDtype(categories=["1", "2", "3", "4", "5"])
    for i in range(1, 27):
        clean_df[f"whoqol_{i}"] = (
            df[f"v1_whoqol_itm{i}"]
            .replace(np.nan, pd.NA)
            .astype("Int64")
            .astype("string")
            .astype(whoqol_cat)
        )

    clean_df["whoqol_total"] = (
        df["v1_whoqol_dom_glob"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["whoqol_phys_health"] = (
        df["v1_whoqol_dom_phys"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["whoqol_psych_health"] = (
        df["v1_whoqol_dom_psy"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["whoqol_soc"] = (
        df["v1_whoqol_dom_soc"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["whoqol_env"] = (
        df["v1_whoqol_dom_env"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )

    big_five_cat = pd.CategoricalDtype(categories=["1", "2", "3", "4", "5"])
    for i in range(1, 11):
        clean_df[f"big_five_{i}"] = (
            df[f"v1_big_five_itm{i}"]
            .replace(np.nan, pd.NA)
            .astype("Int64")
            .astype("string")
            .astype(big_five_cat)
        )
    clean_df["big_five_extraversion"] = (
        df["v1_big_five_extra"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["big_five_neuroticism"] = (
        df["v1_big_five_neuro"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["big_five_conscientiousness"] = (
        df["v1_big_five_consc"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["big_five_openness"] = (
        df["v1_big_five_openn"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
    )
    clean_df["big_five_agreeableness"] = (
        df["v1_big_five_agree"].replace(np.nan, pd.NA).astype(pd.Float32Dtype())
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
