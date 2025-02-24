from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(".").resolve()
ROOT = THIS_DIR.parent.parent.resolve()
DATA_DIR = ROOT / "src" / "data"
BLD_DATA = ROOT / "bld" / "data"
BLD_DATA.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# PHENOTYPIC DATA #
# --------------------------------------------------------------------------------------
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
    df = df.set_index("v1_id")
    clean_df = pd.DataFrame(index=df.index).rename_axis("id", axis="index")
    clean_df["stat"] = df["v1_stat"].astype(
        pd.CategoricalDtype(categories=["CLINICAL", "CONTROL"])
    )
    clean_df["sex"] = df["v1_sex"].astype(pd.CategoricalDtype(categories=["F", "M"]))
    clean_df["age"] = df["v1_age"].astype(pd.Int8Dtype())
    clean_df["seas_birth"] = df["v1_seas_birth"].astype(
        pd.CategoricalDtype(
            categories=["Spring", "Summer", "Fall", "Winter"], ordered=True
        )
    )
    clean_df["age_m_birth"] = df["v1_age_m_birth"].astype(pd.Float32Dtype())
    clean_df["age_f_birth"] = df["v1_age_f_birth"].astype(pd.Float32Dtype())
    clean_df["marital_stat"] = df["v1_marital_stat"].astype(
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
    clean_df["partner"] = _map_yes_no(df["v1_partner"])
    clean_df["no_bio_children"] = df["v1_no_bio_chld"].astype(pd.Float32Dtype())
    clean_df["no_step_children"] = df["v1_stp_chld"].astype(pd.Float32Dtype())
    clean_df["no_adpt_children"] = df["v1_no_adpt_chld"].astype(pd.Float32Dtype())
    clean_df["no_brothers"] = df["v1_brothers"].astype(pd.Float32Dtype())
    clean_df["no_sisters"] = df["v1_sisters"].astype(pd.Float32Dtype())
    clean_df["living_alone"] = _map_yes_no(df["v1_liv_aln"])
    clean_df["school"] = df["v1_school"].astype(pd.Float32Dtype())
    clean_df["ed_status"] = df["v1_ed_status"].astype(pd.Float32Dtype())

    clean_df["employment"] = _map_yes_no(df["v1_curr_paid_empl"])
    clean_df["disability_pension"] = _map_yes_no(df["v1_disabl_pens"])
    clean_df["supported_employment"] = _map_yes_no(df["v1_spec_emp"])
    clean_df["work_absence"] = df["v1_wrk_abs_pst_5_yrs"].astype(pd.Float32Dtype())
    clean_df["work_impairment"] = _map_yes_no(df["v1_cur_work_restr"])
    clean_df["current_psych_treatment"] = df["v1_cur_psy_trm"].astype(pd.Float32Dtype())
    clean_df["outpat_treatment"] = df["v1_outpat_psy_trm"].astype(pd.Float32Dtype())
    clean_df["outpat_treatment_age"] = df["v1_age_1st_out_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["inpat_treatment"] = _map_yes_no(df["v1_daypat_inpat_trm"])
    clean_df["inpat_treatment_age"] = df["v1_age_1st_inpat_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["duration_illness"] = df["v1_dur_illness"].astype(pd.Float32Dtype())
    clean_df["first_ep"] = _map_yes_no_control(df["v1_1st_ep"])
    clean_df["times_treated_inpatient_cont"] = df["v1_tms_daypat_outpat_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["times_treated_inpatient_ord"] = df["v1_cat_daypat_outpat_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["adverse_events_curr_medication"] = _map_yes_no_control(df["v1_adv"])
    clean_df["med_change"] = _map_yes_no_control(df["v1_medchange"])
    clean_df["lithium"] = _map_yes_no_control(df["v1_lith"])
    clean_df["lithium_dur"] = df["v1_lith_prd"].astype(pd.Float32Dtype())
    clean_df["fam_hist"] = _map_yes_no_control(df["v1_fam_hist"])
    clean_df["height"] = df["v1_height"].astype(pd.Float32Dtype())
    clean_df["weight"] = df["v1_weight"].astype(pd.Float32Dtype())
    clean_df["bmi"] = df["v1_bmi"].astype(pd.Float32Dtype())
    clean_df["cholesterol"] = _map_yes_no(df["v1_chol_trig"])
    clean_df["hypertension"] = _map_yes_no(df["v1_hyperten"])
    clean_df["angina_pectoris"] = _map_yes_no(df["v1_ang_pec"])
    clean_df["heartattack"] = _map_yes_no(df["v1_heart_att"])
    clean_df["stroke"] = _map_yes_no(df["v1_stroke"])
    clean_df["diabetes"] = _map_yes_no(df["v1_diabetes"])
    clean_df["hyperthyroid"] = _map_yes_no(df["v1_hyperthy"])
    clean_df["hypothyroid"] = _map_yes_no(df["v1_hypothy"])
    clean_df["osteoporosis"] = _map_yes_no(df["v1_osteopor"])
    clean_df["asthma"] = _map_yes_no(df["v1_asthma"])
    clean_df["copd"] = _map_yes_no(df["v1_copd"])
    clean_df["allergies"] = _map_yes_no(df["v1_allerg"])
    clean_df["neuroderm"] = _map_yes_no(df["v1_neuroder"])
    clean_df["psoriasis"] = _map_yes_no(df["v1_psoriasis"])
    clean_df["autoimm"] = _map_yes_no(df["v1_autoimm"])
    clean_df["cancer"] = _map_yes_no(df["v1_cancer"])
    clean_df["stomach_ulc"] = _map_yes_no(df["v1_stom_ulc"])
    clean_df["kidney_fail"] = _map_yes_no(df["v1_kid_fail"])
    clean_df["stone"] = _map_yes_no(df["v1_stone"])
    clean_df["epilepsy"] = _map_yes_no(df["v1_epilepsy"])
    clean_df["migraine"] = _map_yes_no(df["v1_migraine"])
    clean_df["parkinson"] = _map_yes_no(df["v1_parkinson"])
    clean_df["liv_cir_inf"] = _map_yes_no(df["v1_liv_cir_inf"])
    clean_df["tbi"] = _map_yes_no(df["v1_tbi"])
    clean_df["beh"] = _map_yes_no(df["v1_beh"])
    clean_df["eyear"] = _map_yes_no(df["v1_eyear"])
    clean_df["inf"] = _map_yes_no(df["v1_inf"])

    clean_df["smoker"] = (
        df["v1_ever_smkd"]
        .replace({np.nan: pd.NA, "N": "never", "Y": "yes", "F": "former"})
        .astype(pd.CategoricalDtype(categories=["never", "yes", "former"]))
    )
    clean_df["no_cig"] = df["v1_no_cig"].astype(pd.Float32Dtype())
    clean_df["alc_past_year"] = df["v1_alc_pst12_mths"].astype(pd.Float32Dtype())
    clean_df["alc_5_drinks"] = df["v1_alc_5orm"].astype(pd.Float32Dtype())
    clean_df["alc_dependence"] = _map_yes_no(df["v1_lftm_alc_dep"])
    clean_df["illicit_drugs"] = _map_yes_no(df["v1_evr_ill_drg"])
    clean_df["age_at_mde"] = df["v1_scid_age_MDE"].astype(pd.Float32Dtype())
    clean_df["age_at_mania"] = df["v1_scid_age_mania"].astype(pd.Float32Dtype())
    clean_df["no_of_mania"] = df["v1_scid_no_mania"].astype(pd.Float32Dtype())
    clean_df["age_at_hypomania"] = df["v1_scid_age_hypomania"].astype(pd.Float32Dtype())
    clean_df["no_of_hypomania"] = df["v1_scid_no_hypomania"].astype(pd.Float32Dtype())

    clean_df["ever_delus"] = _map_yes_no_control(df["v1_scid_ever_delus"])
    clean_df["ever_halluc"] = _map_yes_no_control(df["v1_scid_ever_halls"])
    clean_df["ever_psyc"] = _map_yes_no_control(df["v1_scid_ever_psyc"])
    clean_df["ever_suic_ide"] = _map_yes_no_control(df["v1_scid_evr_suic_ide"])
    clean_df["severity_suic_ide"] = df["v1_scid_suic_ide"].astype(pd.Float32Dtype())
    clean_df["suic_methods"] = df["v1_scid_suic_thght_mth"].astype(pd.Float32Dtype())
    clean_df["suic_note"] = df["v1_scid_suic_note_thgts"].astype(pd.Float32Dtype())
    clean_df["suic_attempt"] = df["v1_suic_attmpt"].astype(pd.Float32Dtype())
    clean_df["no_suic_attempt"] = df["v1_scid_no_suic_attmpt"].astype(pd.Float32Dtype())
    clean_df["prep_suic_attempt_ord"] = df["v1_prep_suic_attp_ord"].astype(
        pd.Float32Dtype()
    )
    clean_df["suic_attempt_note"] = df["v1_suic_note_attmpt"].astype(pd.Float32Dtype())

    for i in range(1, 8):
        clean_df[f"panss_p{i}"] = df[f"v1_panss_p{i}"].astype(pd.Float32Dtype())

    clean_df["panss_sum_pos"] = df["v1_panss_sum_pos"].astype(pd.Float32Dtype())
    for i in range(1, 8):
        clean_df[f"panss_n{i}"] = df[f"v1_panss_n{i}"].astype(pd.Float32Dtype())

    clean_df["panss_sum_neg"] = df["v1_panss_sum_neg"].astype(pd.Float32Dtype())

    for i in range(1, 16):
        clean_df[f"panss_g{i}"] = df[f"v1_panss_g{i}"].astype(pd.Float32Dtype())

    clean_df["panss_sum_gen"] = df["v1_panss_sum_gen"].astype(pd.Float32Dtype())
    clean_df["panss_total_score"] = df["v1_panss_sum_tot"].astype(pd.Float32Dtype())

    for i in range(1, 10):
        clean_df[f"idsc_{i}"] = df[f"v1_idsc_itm{i}"].astype(pd.Float32Dtype())

    clean_df["idsc_9a"] = df["v1_idsc_itm9a"].astype(
        pd.CategoricalDtype(categories=["-999", "A", "M", "N"])
    )
    clean_df["idsc_9b"] = _map_yes_no_control(df["v1_idsc_itm9b"])

    for i in range(10, 31):
        clean_df[f"idsc_{i}"] = df[f"v1_idsc_itm{i}"].astype(pd.Float32Dtype())

    clean_df["idsc_total"] = df["v1_idsc_sum"].astype(pd.Float32Dtype())
    for i in range(1, 12):
        clean_df[f"ymrs_{i}"] = df[f"v1_ymrs_itm{i}"].astype(pd.Float32Dtype())

    clean_df["ymrs_total"] = df["v1_ymrs_sum"].astype(pd.Float32Dtype())
    clean_df["cgi"] = df["v1_cgi_s"].astype(pd.Float32Dtype())

    clean_df["gaf"] = df["v1_gaf"].astype(pd.Float32Dtype())
    clean_df["language_skill"] = df["v1_nrpsy_lng"]
    clean_df["tmt_a_time"] = df["v1_nrpsy_tmt_A_rt"].astype(pd.Float32Dtype())
    clean_df["tmt_a_err"] = df["v1_nrpsy_tmt_A_err"].astype(pd.Float32Dtype())
    clean_df["tmt_b_time"] = df["v1_nrpsy_tmt_B_rt"].astype(pd.Float32Dtype())
    clean_df["tmt_b_err"] = df["v1_nrpsy_tmt_B_err"].astype(pd.Float32Dtype())
    clean_df["dgt_sp_fwd"] = df["v1_nrpsy_dgt_sp_frw"].astype(pd.Float32Dtype())
    clean_df["dgt_sp_bck"] = df["v1_nrpsy_dgt_sp_bck"].astype(pd.Float32Dtype())
    clean_df["dst"] = df["v1_nrpsy_dg_sym"].astype(pd.Float32Dtype())
    clean_df["mwtb"] = df["v1_nrpsy_mwtb"].astype(pd.Float32Dtype())
    clean_df["rel_christianity"] = _map_yes_no(df["v1_rel_chr"])
    clean_df["rel_islam"] = _map_yes_no(df["v1_rel_isl"])
    clean_df["rel_other"] = _map_yes_no(df["v1_rel_oth"])
    clean_df["rel_act"] = df["v1_rel_act"].astype(pd.Float32Dtype())
    clean_df["med_compliance_week"] = df["v1_med_pst_wk"].astype(pd.Float32Dtype())

    clean_df["med_compliance_6_months"] = df["v1_med_pst_sx_mths"].astype(
        pd.Float32Dtype()
    )

    for i in (1, 15):
        clean_df[f"whoqol_{i}"] = df[f"v1_whoqol_itm{i}"].astype(pd.Float32Dtype())

    for i in [2] + list(range(16, 27)):
        clean_df[f"whoqol_{i}"] = df[f"v1_whoqol_itm{i}"].astype(pd.Float32Dtype())

    for i in range(3, 15):
        clean_df[f"whoqol_{i}"] = df[f"v1_whoqol_itm{i}"].astype(pd.Float32Dtype())

    clean_df["whoqol_26"] = df["v1_whoqol_itm26"].astype(pd.Float32Dtype())

    clean_df["whoqol_total"] = df["v1_whoqol_dom_glob"].astype(pd.Float32Dtype())
    clean_df["whoqol_phys_health"] = df["v1_whoqol_dom_phys"].astype(pd.Float32Dtype())
    clean_df["whoqol_psych_health"] = df["v1_whoqol_dom_psy"].astype(pd.Float32Dtype())
    clean_df["whoqol_soc"] = df["v1_whoqol_dom_soc"].astype(pd.Float32Dtype())
    clean_df["whoqol_env"] = df["v1_whoqol_dom_env"].astype(pd.Float32Dtype())

    for i in range(1, 11):
        clean_df[f"big_five_{i}"] = df[f"v1_big_five_itm{i}"].astype(pd.Float32Dtype())
    clean_df["big_five_extraversion"] = df["v1_big_five_extra"].astype(
        pd.Float32Dtype()
    )
    clean_df["big_five_neuroticism"] = df["v1_big_five_neuro"].astype(pd.Float32Dtype())
    clean_df["big_five_conscientiousness"] = df["v1_big_five_consc"].astype(
        pd.Float32Dtype()
    )
    clean_df["big_five_openness"] = df["v1_big_five_openn"].astype(pd.Float32Dtype())
    clean_df["big_five_agreeableness"] = df["v1_big_five_agree"].astype(
        pd.Float32Dtype()
    )

    clean_df_clinical = clean_df.query("stat == 'CLINICAL'")
    return clean_df_clinical


def _map_yes_no(sr):
    """Maps the column values to "yes", "no", "pd.NA" """

    mapping = {"N": "no", "Y": "yes", np.nan: pd.NA}

    return sr.map(mapping).astype(pd.CategoricalDtype(categories=["yes", "no"]))


def _map_yes_no_control(sr):
    """Maps the column values to "yes", "no", "pd.NA" """

    mapping = {"-999": "-999", "N": "no", "Y": "yes", np.nan: pd.NA}

    return sr.map(mapping).astype(pd.CategoricalDtype(categories=["yes", "no", "-999"]))


def _map_cat_school(sr):
    school_mapping = {
        0: "no_graduation",
        1: "Hauptschule",
        2: "Realschule_Polytechnischule_Oberschule",
        3: "Allgemeine_Hochschulreife",
        -999: "still_in_school/other",
    }
    dtype = pd.CategoricalDtype(categories=school_mapping.values(), ordered=True)
    return sr.map(school_mapping).astype(dtype)


def _map_cat_psych_treatment(sr):
    mapping = {
        1: "no",
        2: "yes, outpatient",
        3: "yes, day patient",
        4: "yes, inpatient",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_outpat_treatment(sr):
    mapping = {
        1: "no",
        2: "yes_consultation_short_trm",
        3: "yes_cont_trm_six_months_multiple_short_ep",
        4: "yes_cont_trm_years_many_short_ep",
        -999: "no_info",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_times_treated_inpatient(sr):
    mapping = {
        1: "smaller_5_times",
        2: "6-10_times",
        3: "11-14_times",
        4: "15_times_or_more",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_lith_dur(sr):
    mapping = {
        -999: "never_or_control",
        1: "less_than_1_year",
        2: "1-2_years",
        3: "2_years_or_more",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_alc_past_year(sr):
    mapping = {
        1: "never",
        2: "only_on_special_occacions",
        3: "once_per_month",
        4: "2-4_times_month",
        5: "2-3_times_week",
        6: "4_times_week",
        7: "daily",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_alc_5_drinks(sr):
    mapping = {
        -999: "skipped_irregular",
        1: "never",
        2: "1-2_times",
        3: "3-5_times",
        4: "6-11_times",
        5: "1_times_month",
        6: "2-3_times_month",
        7: "1-2_times_week",
        8: "3-4_times_week",
        9: "daily",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_severity_suic_ide(sr):
    mapping = {
        -999: "skipped",
        1: "only_fleeting",
        2: "serious_thoughts",
        3: "persistent_thoughts",
        4: "serious_and_persistent_thoughts",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)

    return sr.map(mapping).astype(dtype)


def _map_cat_suic_methods(sr):
    mapping = {
        -999: "skipped",
        1: "no",
        2: "yes_without_details",
        3: "yes_with_details",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_suic_note(sr):
    mapping = {
        -999: "skipped",
        1: "no",
        2: "thought_about",
        3: "persistent_thoughts",
        4: "thought_about_and_persistent_thoughts",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_suic_attempt(sr):
    mapping = {-999: "skipped", 1: "no", 2: "interruption_of_attempt", 3: "yes"}
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_no_suic_attempt(sr):
    mapping = {
        -999: "skipped",
        1: "1_time",
        2: "2_times",
        3: "3_times",
        4: "4_times",
        5: "5_times",
        6: "6_or_more_times",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_prep_suic_attempt_ord(sr):
    mapping = {
        -999: "skipped",
        1: "no_prep",
        2: "little_prep",
        3: "moderate_prep",
        4: "extensive_prep",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_panss(sr):
    mapping = {
        1: "absent",
        2: "minimal",
        3: "mild",
        4: "moderate",
        5: "moderate severe",
        6: "severe",
        7: "extreme",
    }

    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_rel_act(sr):
    mapping = {
        1: "not_at_all",
        2: "little_active",
        3: "moderately_active",
        4: "rather_active",
        5: "very_active",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_med_compliance(sr):
    mapping = {
        -999: "control",
        1: "every_day_as_prescribed",
        2: "every_day_but_not_as_prescribed",
        3: "regularly_but_not_every_day",
        4: "sometimes_but_not_regularly",
        5: "seldom",
        6: "not_at_all",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_whoqol_1_and_15(sr):
    mapping = {
        1: "very_poor",
        2: "poor",
        3: "neither_poor_nor_good",
        4: "good",
        5: "very_good",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_whoqol_2_and_16_25(sr):
    mapping = {
        1: "very_dissatisfied",
        2: "dissatisfied",
        3: "neither_dis_nor_satisfied",
        4: "satisfied",
        5: "very_satisfied",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_whoqol_3_14(sr):
    mapping = {
        1: "not_at_all",
        2: "a_little",
        3: "moderate_amount",
        4: "very_much",
        5: "an_extreme_amount",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_whoqol_26(sr):
    mapping = {1: "never", 2: "seldom", 3: "quite_often", 4: "very_often", 5: "always"}
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


def _map_cat_big_five(sr):
    mapping = {
        1: "disagree_strongly",
        2: "disagree_little",
        3: "neither_dis_nor_agree",
        4: "agree_little",
        5: "agree_strongly",
    }
    dtype = pd.CategoricalDtype(categories=mapping.values(), ordered=True)
    return sr.map(mapping).astype(dtype)


# --------------------------------------------------------------------------------------
# LIPIDOMIC DATA
# --------------------------------------------------------------------------------------


def clean_lipidomic_data(sample_description, lipid_intensities):
    """Cleans the lipidomic data. Takes the lipid_intensities file and the
    sample description file; removes duplicates and removes the lipids that are affected
    by fasting status according to Tkachev et al., 2023

    Args:
        df (pd.DataFrame): The input dataframe containing raw lipidomic data.
        sample_desc (pd.DataFrame): The input dataframe containing sample descriptions.

    Returns:
        pd.DataFrame: The cleaned lipidomic dataframe.

    """
    clean_sample_description = _clean_sample_description(sample_description)
    clean_lipid_intensities = _clean_lipid_intensities(lipid_intensities)
    merged_df = clean_sample_description.join(clean_lipid_intensities, on="ind")

    return merged_df


def _clean_sample_description(df):
    """Takes sample description, sets index and drops unnecessary columns."""

    clean_sample_description = df.drop_duplicates(subset="Patient_ID")
    clean_sample_description = clean_sample_description.set_index(["Patient_ID", "ind"])
    clean_sample_description["age"] = clean_sample_description["age"].astype(
        pd.Int8Dtype()
    )
    clean_sample_description["sex"] = clean_sample_description["sex"].astype(
        pd.CategoricalDtype()
    )
    unnecessary_columns = [
        "bmi",
        "diagnosis",
        "STARLIMS_sic",
        "clinic",
        "year",
        "repeated visit (delete)",
    ]
    clean_sample_description = clean_sample_description.drop(
        columns=unnecessary_columns
    )
    return clean_sample_description


def _clean_lipid_intensities(df):
    """Takes the lipid intensities and sets the correct index."""

    clean_lipid_intensities = df.set_index("originalMS#").rename_axis(
        "ind", axis="rows"
    )
    return clean_lipid_intensities


if __name__ == "__main__":
    df = pd.read_csv(
        DATA_DIR / "230614_v6.0" / "230614_v6.0_psycourse_wd.csv", delimiter="\t"
    )
    clean_df = clean_phenotypic_data(df)
    clean_df.to_csv(BLD_DATA / "clean_phenotypic_data.csv")
    clean_df.to_pickle(BLD_DATA / "clean_phenotypic_data.pkl")
