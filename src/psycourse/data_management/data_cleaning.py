import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------------------
# PHENOTYPIC DATA #
# --------------------------------------------------------------------------------------
def clean_phenotypic_data(df):
    """Cleans the phenotypic data from '230614_v6.0_psycourse_wd.csv' by:

    - Selecting relevant columns based on criteria analogous to Dwyer et al. (2020).
    - Renaming columns for clarity.
    - Setting appropriate data types. Note: For now, -999 are turned into NAs.

    Args:
        df (pd.DataFrame): The input dataframe containing raw phenotypic data.

    Returns:
        pd.DataFrame: The cleaned dataframe with selected and formatted columns.

    """
    df = df.set_index("v1_id")
    df = df.query(
        "v1_stat == 'CLINICAL'"
    )  # raw_data only contains clinical patients now
    clean_df = pd.DataFrame(index=df.index).rename_axis("id", axis="index")
    clean_df["sex"] = df["v1_sex"].astype(pd.CategoricalDtype(categories=["F", "M"]))
    clean_df["age"] = df["v1_age"].astype(pd.Int8Dtype())
    clean_df["diagnosis"] = df["v1_scid_dsm_dx"].astype(pd.CategoricalDtype())
    clean_df["diagnosis_sum"] = df["v1_scid_dsm_dx_cat"].astype(pd.CategoricalDtype())
    clean_df["gsa_id"] = df["gsa_id"]
    clean_df["seas_birth"] = df["v1_seas_birth"].astype(
        pd.CategoricalDtype(
            categories=["Spring", "Summer", "Fall", "Winter"], ordered=True
        )
    )
    clean_df["age_m_birth"] = df["v1_age_m_birth"].astype(pd.Int32Dtype())
    clean_df["age_f_birth"] = df["v1_age_f_birth"].astype(pd.Int8Dtype())
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
    clean_df["no_bio_children"] = df["v1_no_bio_chld"].astype(pd.Int8Dtype())
    clean_df["no_step_children"] = df["v1_stp_chld"].astype(pd.Int8Dtype())
    clean_df["no_adpt_children"] = df["v1_no_adpt_chld"].astype(pd.Int8Dtype())
    clean_df["no_brothers"] = df["v1_brothers"].astype(pd.Int8Dtype())
    clean_df["no_sisters"] = df["v1_sisters"].astype(pd.Int8Dtype())
    clean_df["living_alone"] = _map_yes_no(df["v1_liv_aln"])
    clean_df["school"] = _bump_ordinal(df["v1_school"])
    clean_df["ed_status"] = _bump_ordinal(df["v1_ed_status"])

    clean_df["employment"] = _map_yes_no(df["v1_curr_paid_empl"])
    clean_df["disability_pension"] = _map_yes_no(df["v1_disabl_pens"])
    clean_df["supported_employment"] = _map_yes_no(df["v1_spec_emp"])
    # clean_df["work_absence"] = df["v1_wrk_abs_pst_5_yrs"].astype(pd.Int8Dtype())
    clean_df["work_impairment"] = _map_yes_no(df["v1_cur_work_restr"])
    clean_df["current_psych_treatment"] = df["v1_cur_psy_trm"].astype(pd.Float32Dtype())
    clean_df["outpat_treatment"] = _replace_999_with_NA(df["v1_outpat_psy_trm"]).astype(
        pd.Int8Dtype()
    )
    clean_df["outpat_treatment_age"] = df["v1_age_1st_out_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["inpat_treatment"] = _map_yes_no(df["v1_daypat_inpat_trm"])
    clean_df["inpat_treatment_age"] = df["v1_age_1st_inpat_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["duration_illness"] = df["v1_dur_illness"].astype(pd.Float32Dtype())
    clean_df["first_ep"] = _map_yes_no(df["v1_1st_ep"])
    clean_df["times_treated_inpatient_cont"] = df["v1_tms_daypat_outpat_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["times_treated_inpatient_ord"] = df["v1_cat_daypat_outpat_trm"].astype(
        pd.Float32Dtype()
    )
    clean_df["adverse_events_curr_medication"] = _map_yes_no(df["v1_adv"])
    clean_df["med_change"] = _map_yes_no(df["v1_medchange"])
    clean_df["lithium"] = _map_yes_no(df["v1_lith"])
    clean_df["lithium_dur"] = _replace_999_with_0(df["v1_lith_prd"]).astype(
        pd.Int8Dtype()
    )
    clean_df["fam_hist"] = _map_yes_no(df["v1_fam_hist"])  # -999 -> NA
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
    clean_df["no_cig"] = _replace_999_with_0(df["v1_no_cig"]).astype(pd.Int64Dtype())
    clean_df["alc_past_year"] = df["v1_alc_pst12_mths"].astype(pd.Float32Dtype())
    clean_df["alc_5_drinks"] = _replace_999_with_0(df["v1_alc_5orm"]).astype(
        pd.Int8Dtype()
    )
    clean_df["alc_dependence"] = _map_yes_no(df["v1_lftm_alc_dep"])
    clean_df["illicit_drugs"] = _map_yes_no(df["v1_evr_ill_drg"])
    clean_df["age_at_mde"] = df["v1_scid_age_MDE"].astype(pd.Float32Dtype())
    clean_df["age_at_mania"] = df["v1_scid_age_mania"].astype(pd.Float32Dtype())
    clean_df["no_of_mania"] = _map_999_with_0_99_to_NA(df["v1_scid_no_mania"])
    clean_df["age_at_hypomania"] = df["v1_scid_age_hypomania"].astype(pd.Float32Dtype())
    clean_df["no_of_hypomania"] = _map_999_with_0_99_to_NA(
        df["v1_scid_no_hypomania"]
    ).astype(pd.Float32Dtype())

    clean_df["ever_delus"] = _map_yes_no(df["v1_scid_ever_delus"])
    clean_df["ever_halluc"] = _map_yes_no(df["v1_scid_ever_halls"])
    clean_df["ever_psyc"] = _map_yes_no(df["v1_scid_ever_psyc"])
    clean_df["ever_suic_ide"] = _map_yes_no(df["v1_scid_evr_suic_ide"])
    clean_df["severity_suic_ide"] = _replace_999_with_0(df["v1_scid_suic_ide"]).astype(
        pd.Int8Dtype()
    )
    clean_df["suic_methods"] = _replace_999_with_0(df["v1_scid_suic_thght_mth"]).astype(
        pd.Int8Dtype()
    )
    clean_df["suic_note"] = _replace_999_with_0(df["v1_scid_suic_note_thgts"]).astype(
        pd.Int8Dtype()
    )
    clean_df["suic_attempt"] = _replace_999_with_0(df["v1_suic_attmpt"]).astype(
        pd.Int8Dtype()
    )
    clean_df["no_suic_attempt"] = _replace_999_with_0(
        df["v1_scid_no_suic_attmpt"]
    ).astype(pd.Int8Dtype())
    clean_df["prep_suic_attempt_ord"] = _replace_999_with_0(
        df["v1_prep_suic_attp_ord"]
    ).astype(pd.Int8Dtype())
    clean_df["suic_attempt_note"] = _replace_999_with_0(
        df["v1_suic_note_attmpt"]
    ).astype(pd.Int8Dtype())

    for i in range(1, 8):
        clean_df[f"panss_p{i}"] = df[f"v1_panss_p{i}"].astype(pd.Int8Dtype())

    clean_df["panss_sum_pos"] = df["v1_panss_sum_pos"].astype(pd.Int8Dtype())
    for i in range(1, 8):
        clean_df[f"panss_n{i}"] = df[f"v1_panss_n{i}"].astype(pd.Int8Dtype())

    clean_df["panss_sum_neg"] = df["v1_panss_sum_neg"].astype(pd.Float32Dtype())

    for i in range(1, 16):
        clean_df[f"panss_g{i}"] = df[f"v1_panss_g{i}"].astype(pd.Int8Dtype())

    clean_df["panss_sum_gen"] = df["v1_panss_sum_gen"].astype(pd.Float32Dtype())
    clean_df["panss_total_score"] = df["v1_panss_sum_tot"].astype(pd.Float32Dtype())

    for i in range(1, 31):
        clean_df[f"idsc_{i}"] = df[f"v1_idsc_itm{i}"].astype(pd.Float32Dtype())

    clean_df["idsc_total"] = df["v1_idsc_sum"].astype(pd.Float32Dtype())
    for i in range(1, 12):
        clean_df[f"ymrs_{i}"] = df[f"v1_ymrs_itm{i}"].astype(pd.Int8Dtype())

    clean_df["ymrs_total"] = df["v1_ymrs_sum"].astype(pd.Float32Dtype())
    clean_df["cgi"] = _replace_999_with_NA(df["v1_cgi_s"]).astype(pd.Int8Dtype())

    clean_df["gaf"] = df["v1_gaf"].astype(pd.Float32Dtype())
    clean_df["language_skill"] = df["v1_nrpsy_lng"].astype(
        pd.CategoricalDtype(
            categories=["not sufficient", "sufficient", "good", "mother tongue"],
            ordered=True,
        )
    )
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
    # # zero variance and messes up
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
        clean_df[f"big_five_{i}"] = df[f"v1_big_five_itm{i}"].astype(pd.Int8Dtype())
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

    return clean_df


def _map_yes_no(sr):
    """Maps the column values to "yes", "no", "pd.NA" """

    mapping = {"N": "no", "Y": "yes", np.nan: pd.NA}

    return sr.map(mapping).astype(pd.CategoricalDtype(categories=["yes", "no"]))


def _replace_999_with_NA(sr):
    """Replaces -999 with pd.NA in the series."""

    return sr.replace(-999, np.nan)


def _replace_999_with_0(sr):
    """Replaces -999 with 0 in the series."""

    return sr.replace(-999, 0)


def _map_999_with_0_99_to_NA(sr):
    """Replaces -999 with 0 in the series."""

    mapping = {-999: 0, 99: np.nan}
    return sr.map(mapping).astype(pd.Float32Dtype())


def _bump_ordinal(sr):
    """Bumps the ordinal values by 1, -999 to 0."""
    # Apply the transformation: if value == 0, assign -999, else add 1.
    bumped_transformed = sr.apply(lambda x: 0 if x == -999 else x + 1)
    # Convert to a nullable integer type
    return bumped_transformed.astype(pd.Int8Dtype())


# --------------------------------------------------------------------------------------
# LIPIDOMIC DATA
# --------------------------------------------------------------------------------------


def clean_lipidomic_data(sample_description, lipid_intensities):
    """Cleans the lipidomic data. Takes the lipid_intensities file and the
    sample description file; removes duplicates and removes the lipids that are affected
    by fasting status according to Tkachev et al., 2023 or have a skewed distribution
    according to analysis by colleagues in Munich (did not check myself).

    Args:
        sample_description(pd.DataFrame): The input dataframe containing
                                        sample descriptions.
        lipid_intensities(pd.DataFrame): The input dataframe containing
                                        raw lipidomic data.

    Returns:
        pd.DataFrame: The cleaned lipidomic dataframe.

    """
    clean_sample_description = _clean_sample_description(sample_description)
    clean_lipid_intensities = _clean_lipid_intensities(lipid_intensities)
    merged_df = clean_sample_description.join(clean_lipid_intensities, on="ind")
    merged_df = merged_df.set_index("Patient_ID").rename_axis("id", axis="index")

    return merged_df


def _clean_sample_description(df):
    """Takes sample description, sets index and drops unnecessary columns."""

    clean_sample_description = df.drop_duplicates(subset="Patient_ID")
    clean_sample_description = clean_sample_description.set_index("ind")
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
    """Takes the lipid intensities and sets the correct index, only keeps lipids that
    are annotated, not affected by fasting status and not skewed."""

    clean_lipid_intensities = df.set_index("originalMS#").rename_axis(
        "ind", axis="rows"
    )

    clean_lipid_intensities = _remove_non_annotated_lipids(clean_lipid_intensities)
    clean_lipid_intensities = _remove_skewed_lipids(clean_lipid_intensities)
    clean_lipid_intensities = _remove_fasting_lipids(clean_lipid_intensities)
    clean_lipid_intensities = _remove_outliers(clean_lipid_intensities)

    return clean_lipid_intensities


def _remove_non_annotated_lipids(df):
    """Removes lipids that are not annotated in the annotated_lipids list"""
    annotated_lipids = [
        "gpeakpos7470",
        "gpeakpos7514",
        "gpeakpos7730",
        "gpeakpos7769",
        "gpeakpos7961",
        "gpeakpos7997",
        "gpeakpos8043",
        "gpeakpos8324",
        "gpeakpos8631",
        "gpeakpos8679",
        "gpeakpos8733",
        "gpeakpos9000",
        "gpeakpos9264",
        "gpeakpos9328",
        "gpeakpos9390",
        "gpeakpos9446",
        "gpeakpos9505",
        "gpeakpos9794",
        "gpeakpos9863",
        "gpeakpos9918",
        "gpeakpos10107",
        "gpeakpos10171",
        "gpeakpos10236",
        "gpeakpos10371",
        "gpeakpos10445",
        "gpeakpos10660",
        "gpeakpos10723",
        "gpeakpos10860",
        "gpeakpos10926",
        "gpeakpos11003",
        "gpeakpos11076",
        "gpeakpos11134",
        "gpeakpos11200",
        "gpeakpos11253",
        "gpeakpos11314",
        "gpeakpos11389",
        "gpeakpos11522",
        "gpeakpos11585",
        "gpeakpos11639",
        "gpeakpos11690",
        "gpeakpos11823",
        "gpeakpos11828",
        "gpeakpos11905",
        "gpeakpos11964",
        "gpeakpos12022",
        "gpeakpos12080",
        "gpeakpos12125",
        "gpeakpos12192",
        "gpeakpos12323",
        "gpeakpos12430",
        "gpeakpos12484",
        "gpeakpos12743",
        "gpeakpos12792",
        "gpeakpos12841",
        "gpeakpos12892",
        "gpeakpos12955",
        "gpeakpos13012",
        "gpeakpos13066",
        "gpeakpos13118",
        "gpeakpos13174",
        "gpeakpos13342",
        "gpeakpos13459",
        "gpeakpos13500",
        "gpeakpos13549",
        "gpeakpos13594",
        "gpeakpos13728",
        "gpeakpos13775",
        "gpeakpos13980",
        "gpeakpos14026",
        "gpeakpos14078",
        "gpeakpos14294",
        "gpeakpos1999",
        "gpeakpos2041",
        "gpeakpos2069",
        "gpeakpos2099",
        "gpeakpos2229",
        "gpeakpos2348",
        "gpeakpos2417",
        "gpeakpos2561",
        "gpeakpos2582",
        "gpeakpos2688",
        "gpeakpos2726",
        "gpeakpos2767",
        "gpeakpos3064",
        "gpeakpos3079",
        "gpeakpos3098",
        "gpeakpos3375",
        "gpeakpos3419",
        "gpeakpos3441",
        "gpeakpos6622",
        "gpeakpos6657",
        "gpeakpos7122",
        "gpeakpos7158",
        "gpeakpos7549",
        "gpeakpos7586",
        "gpeakpos4442",
        "gpeakpos4173",
        "gpeakpos4188",
        "gpeakpos4465",
        "gpeakpos4494",
        "gpeakpos5151",
        "gpeakpos5531",
        "gpeakpos5562",
        "gpeakpos5917",
        "gpeakneg4189",
        "gpeakneg4215",
        "gpeakneg4417",
        "gpeakneg4439",
        "gpeakneg4457",
        "gpeakneg4477",
        "gpeakneg4581",
        "gpeakneg4632",
        "gpeakneg4687",
        "gpeakneg4707",
        "gpeakneg4783",
        "gpeakneg4835",
        "gpeakneg4867",
        "gpeakneg4889",
        "gpeakneg4888",
        "gpeakneg4907",
        "gpeakneg5123",
        "gpeakneg5161",
        "gpeakneg5189",
        "gpeakneg5363",
        "gpeakneg5385",
        "gpeakneg5503",
        "gpeakneg5513",
        "gpeakneg5524",
        "gpeakneg5547",
        "gpeakneg5647",
        "gpeakneg4244",
        "gpeakneg4456",
        "gpeakneg4476",
        "gpeakneg4501",
        "gpeakneg4671",
        "gpeakneg4706",
        "gpeakneg4724",
        "gpeakneg4798",
        "gpeakneg4906",
        "gpeakneg4920",
        "gpeakneg4939",
        "gpeakneg4960",
        "gpeakneg5145",
        "gpeakneg5173",
        "gpeakneg5174",
        "gpeakneg5190",
        "gpeakneg5191",
        "gpeakneg5209",
        "gpeakneg5352",
        "gpeakneg5361",
        "gpeakneg5373",
        "gpeakneg5386",
        "gpeakneg5400",
        "gpeakneg5416",
        "gpeakneg5502",
        "gpeakneg5512",
        "gpeakneg5546",
        "gpeakneg5660",
        "gpeakneg5675",
        "gpeakneg5691",
        "gpeakneg5701",
        "gpeakneg2637",
        "gpeakneg2747",
        "gpeakneg2765",
        "gpeakneg2886",
        "gpeakneg2895",
        "gpeakneg3053",
        "gpeakneg3252",
        "gpeakneg3264",
        "gpeakneg3276",
        "gpeakneg3402",
        "gpeakneg3514",
        "gpeakneg3531",
        "gpeakneg3543",
        "gpeakneg3566",
        "gpeakneg3666",
        "gpeakneg3685",
        "gpeakneg3687",
        "gpeakneg3714",
        "gpeakneg3839",
        "gpeakneg3840",
        "gpeakneg4081",
        "gpeakneg4106",
        "gpeakneg4323",
        "gpeakneg4342",
        "gpeakneg4363",
        "gpeakneg4454",
        "gpeakneg4472",
        "gpeakneg4494",
        "gpeakneg4563",
        "gpeakneg4579",
        "gpeakneg4580",
        "gpeakneg4593",
        "gpeakneg4606",
        "gpeakneg4623",
        "gpeakneg4667",
        "gpeakneg4684",
        "gpeakneg4702",
        "gpeakneg4721",
        "gpeakneg4781",
        "gpeakneg4782",
        "gpeakneg4796",
        "gpeakneg4797",
        "gpeakneg4811",
        "gpeakneg4831",
        "gpeakneg4849",
        "gpeakneg4872",
        "gpeakneg4904",
        "gpeakneg4958",
        "gpeakneg5040",
        "gpeakneg5055",
        "gpeakneg5056",
        "gpeakneg5079",
        "gpeakneg5106",
        "gpeakneg5107",
        "gpeakneg5127",
        "gpeakneg5187",
        "gpeakneg5207",
        "gpeakneg5292",
        "gpeakneg5307",
        "gpeakneg5322",
        "gpeakneg5323",
        "gpeakneg5336",
        "gpeakneg5410",
        "gpeakneg5450",
        "gpeakneg5459",
        "gpeakneg5470",
        "gpeakneg3534",
        "gpeakneg3819",
        "gpeakneg3846",
        "gpeakneg3944",
        "gpeakneg3966",
        "gpeakneg4047",
        "gpeakneg4062",
        "gpeakneg4077",
        "gpeakneg4181",
        "gpeakneg4200",
        "gpeakneg4292",
        "gpeakneg4304",
        "gpeakneg4319",
        "gpeakneg4333",
        "gpeakneg4448",
        "gpeakneg4574",
        "gpeakneg4588",
        "gpeakneg4698",
        "gpeakneg4715",
        "gpeakneg4805",
        "gpeakneg4821",
        "gpeakneg4929",
        "gpeakneg4953",
        "gpeakneg5034",
        "gpeakneg5049",
        "gpeakneg5070",
        "gpeakneg5093",
        "gpeakneg5218",
        "gpeakneg5285",
        "gpeakneg5302",
        "gpeakneg5316",
        "gpeakneg5331",
        "gpeakneg163",
        "gpeakneg291",
        "gpeakneg300",
        "gpeakneg305",
        "gpeakneg349",
        "gpeakneg372",
        "gpeakneg378",
        "gpeakneg384",
        "gpeakneg391",
        "gpeakneg437",
        "gpeakneg451",
        "gpeakneg574",
        "gpeakneg588",
        "gpeakneg596",
        "gpeakneg731",
        "gpeakneg828",
        "gpeakneg838",
        "gpeakneg852",
        "gpeakneg892",
        "gpeakneg902",
        "gpeakneg910",
        "gpeakneg925",
        "gpeakneg935",
        "gpeakneg940",
        "gpeakneg954",
        "gpeakneg1109",
        "gpeakneg1173",
        "gpeakneg1180",
        "gpeakneg1214",
        "gpeakneg1228",
        "gpeakneg1241",
        "gpeakneg1346",
        "gpeakneg1373",
        "gpeakneg1382",
        "gpeakneg1395",
        "gpeakneg1403",
        "gpeakneg1412",
        "gpeakneg1472",
        "gpeakneg1488",
        "gpeakneg1506",
        "gpeakneg1516",
        "gpeakneg1533",
        "gpeakneg1541",
        "gpeakneg1580",
        "gpeakneg1638",
        "gpeakneg1645",
        "gpeakneg1652",
        "gpeakneg1661",
        "gpeakneg1677",
        "gpeakneg1699",
        "gpeakneg1723",
        "gpeakneg1722",
        "gpeakneg1737",
        "gpeakneg1778",
        "gpeakneg1793",
        "gpeakneg1860",
        "gpeakneg3735",
        "gpeakneg3953",
        "gpeakneg4151",
        "gpeakneg4172",
        "gpeakneg4669",
        "gpeakneg3463",
        "gpeakneg3495",
        "gpeakneg3609",
        "gpeakneg3680",
        "gpeakneg3706",
        "gpeakneg3762",
        "gpeakneg3824",
        "gpeakneg3836",
        "gpeakneg3855",
        "gpeakneg3952",
        "gpeakneg3974",
        "gpeakneg3993",
        "gpeakneg3994",
        "gpeakneg4014",
        "gpeakneg4069",
        "gpeakneg4187",
        "gpeakneg4210",
        "gpeakneg4239",
        "gpeakneg4438",
        "gpeakneg4474",
        "gpeakneg4685",
        "gpeakneg2026",
        "gpeakneg2039",
        "gpeakneg2174",
        "gpeakneg2200",
        "gpeakneg2206",
        "gpeakneg2319",
        "gpeakneg2333",
        "gpeakneg2339",
        "gpeakneg2428",
        "gpeakneg2429",
        "gpeakneg2515",
        "gpeakneg2552",
        "gpeakneg2559",
        "gpeakneg2635",
        "gpeakneg2660",
        "gpeakneg2668",
        "gpeakneg2680",
        "gpeakneg2698",
        "gpeakneg2744",
        "gpeakneg2762",
        "gpeakneg2787",
        "gpeakneg2795",
        "gpeakneg2811",
        "gpeakneg2822",
        "gpeakneg2832",
        "gpeakneg2900",
        "gpeakneg2914",
        "gpeakneg2927",
        "gpeakneg3029",
        "gpeakneg3071",
        "gpeakneg3239",
        "gpeakneg3247",
        "gpeakneg3257",
        "gpeakneg3268",
        "gpeakneg3285",
        "gpeakneg3625",
        "gpeakneg3647",
        "gpeakneg3786",
        "gpeakneg3852",
        "gpeakneg3872",
        "gpeakneg3892",
        "gpeakneg3913",
        "gpeakneg3935",
        "gpeakneg4012",
        "gpeakneg4079",
        "gpeakneg4101",
        "gpeakneg4124",
        "gpeakneg4141",
        "gpeakneg4321",
        "gpeakneg4339",
        "gpeakneg4359",
        "gpeakneg4360",
        "gpeakneg4380",
    ]
    return df.drop(columns=set(df.columns) - set(annotated_lipids), errors="ignore")


def _remove_skewed_lipids(df):
    """Removes lipids that have a skewed distribution"""

    skewed_lipids = [
        "gpeakpos6496",
        "gpeakpos6657",
        "gpeakpos6714",
        "gpeakpos7639",
        "gpeakpos7939",
        "gpeakpos9077",
        "gpeakpos9083",
        "gpeakpos9393",
        "gpeakpos9821",
        "gpeakpos10301",
        "gpeakpos12417",
        "gpeakpos12493",
        "gpeakpos15263",
        "gpeakpos15939",
        "gpeakpos16376",
        "gpeakpos16716",
        "gpeakpos16827",
        "gpeakpos17328",
        "gpeakpos17376",
        "gpeakpos17381",
        "gpeakneg163",
        "gpeakneg305",
        "gpeakneg1022",
        "gpeakneg1010",
        "gpeakneg2153",
        "gpeakneg3930",
        "gpeakneg4662",
        "gpeakneg5486",
        "gpeakneg5826",
        "gpeakneg5907",
        "gpeakneg5989",
    ]

    df = df.drop(columns=skewed_lipids, errors="ignore")

    return df


def _remove_fasting_lipids(df):
    lipids_affected_by_fasting = [
        "gpeakneg574",
        "gpeakneg588",
        "gpeakneg596",
        "gpeakneg828",
        "gpeakneg838",
        "gpeakneg852",
        "gpeakneg892",
        "gpeakneg902",
        "gpeakneg910",
        "gpeakneg925",
        "gpeakneg935",
        "gpeakneg940",
        "gpeakneg954",
        "gpeakpos3079",
        "gpeakpos4173",
        "gpeakneg5352",
        "gpeakneg4796",
        "gpeakneg384",
        "gpeakneg349",
        "gpeakneg1506",
        "gpeakneg1373",
        "gpeakneg1382",
        "gpeakneg1395",
        "gpeakneg1403",
        "gpeakneg1412",
        "gpeakneg1180",
        "gpeakneg1214",
        "gpeakneg1228",
        "gpeakneg1241",
        "gpeakneg1109",
    ]

    return df.drop(columns=lipids_affected_by_fasting, errors="ignore")


def _remove_outliers(lipid_data):
    """Remove outliers from lipid data using Z-score method.
    Args:
        lipid_data (pd.DataFrame): DataFrame containing lipidomic data.
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    threshold = 3
    cleaned_lipids = lipid_data.copy()
    z_scores = cleaned_lipids.iloc[:, 2:].apply(stats.zscore)
    outlier_mask = np.abs(z_scores) > threshold
    cleaned_lipids.iloc[:, 2:] = cleaned_lipids.iloc[:, 2:].mask(outlier_mask, np.nan)

    return cleaned_lipids


######################################################################################
# LIPIDOMIC CLASS DATA
######################################################################################


def clean_lipidomic_class_label(annotation_data):
    """Takes the annotation.csv and makes a pd.DataFrame containing lipid class info.
    Args:
        annotation_data (pd.DataFrame):
        The input dataframe with lipid class annotations
    Returns:
        pd.DataFrame: The cleaned lipid class dataframe."""

    annotation_data = annotation_data.set_index("GER").rename_axis("lipid", axis="rows")
    clean_annotation_data = pd.DataFrame(index=annotation_data.index)

    clean_annotation_data["class"] = annotation_data["Lipid class"].astype(
        pd.CategoricalDtype()
    )
    clean_annotation_data["lipid_species"] = annotation_data["Lipid species"]

    return clean_annotation_data


######################################################################################
# CLUSTER LABELS
######################################################################################


def clean_cluster_labels(labels_df):
    """Cleans the cluster labels dataframe. Sets ind; removes unnecessary columns."""

    labels_df = labels_df.set_index("cases").rename_axis("ind", axis="rows")
    cleaned_labels = pd.DataFrame(index=labels_df.index)
    cleaned_labels["cluster_label"] = labels_df["cluster_label"]

    return cleaned_labels


########################################################################################
# PRS DATA
########################################################################################


def clean_prs_data(prs_data, bpd_data, lipid_bpd_data, lipid_scz_data, lipid_mdd_data):
    prs_data = prs_data.set_index(prs_data["FID"])
    clean_prs_df = pd.DataFrame(index=(prs_data["FID"]).rename("gsa_id"))
    prs_renaming_dict = {
        "ADHD_Demontis2023_pst_eff_a1_b0.5_phiauto": "ADHD_PRS",
        "ASD_Grove2019_pst_eff_a1_b0.5_phiauto": "ASD_PRS",
        "EduYears_Okbay2022_pst_eff_a1_b0.5_phiauto": "Education_PRS",
        "SCZ_core_Trubetskoy2022_pst_eff_a1_b0.5_phiauto": "SCZ_PRS",
        "agreeableness_Gupta2024_pst_eff_a1_b0.5_phiauto": "Agreeableness_PRS",
        "alzheimer_bellenguez2022_pst_eff_a1_b0.5_phiauto": "Alzheimer_PRS",
        "conscientiousness_Gupta2024_pst_eff_a1_b0.5_phiauto": "Conscientiousness_PRS",
        "extraversion_Gupta2024_pst_eff_a1_b0.5_phiauto": "Extraversion_PRS",
        "mdd_adams2025_pst_eff_a1_b0.5_phiauto": "MDD_PRS",
        "neuroticism_Gupta2024_pst_eff_a1_b0.5_phiauto": "Neuroticism_PRS",
        "openness_Gupta2024_pst_eff_a1_b0.5_phiauto": "Openness_PRS",
        "sleep_disturbance_long_austinzimmerman2023_pst_eff_a1_b0.5_phiauto": "SleepDurationLong_PRS",  # noqa: E501
        "sleep_disturbance_short_austinzimmerman2023_pst_eff_a1_b0.5_phiauto": "SleepDurationShort_PRS",  # noqa: E501
    }
    for key, value in prs_renaming_dict.items():
        if key in prs_data.columns:
            clean_prs_df[value] = prs_data[key].astype(pd.Float64Dtype())
        else:
            print(f"Warning: {key} not found in prs_data columns.")

    bpd_data = bpd_data.set_index(bpd_data["FID"])
    clean_prs_df["BD_PRS"] = bpd_data[
        "BIP_OConnell2025_no_psycourse_pst_eff_a1_b0.5_phiauto"
    ].astype(pd.Float64Dtype())

    lipid_bpd_data = lipid_bpd_data.set_index(lipid_bpd_data["FID"])
    clean_prs_df["Lipid_BD_PRS"] = lipid_bpd_data[
        "BIP_OConnell2025_no_psycourse_pst_eff_a1_b0.5_phiauto"
    ].astype(pd.Float64Dtype())

    lipid_scz_data = lipid_scz_data.set_index(lipid_scz_data["FID"])
    clean_prs_df["Lipid_SCZ_PRS"] = lipid_scz_data[
        "SCZ_core_Trubetskoy2022_pst_eff_a1_b0.5_phiauto"
    ].astype(pd.Float64Dtype())

    lipid_mdd_data = lipid_mdd_data.set_index(lipid_mdd_data["FID"])
    clean_prs_df["Lipid_MDD_PRS"] = lipid_mdd_data[
        "mdd_adams2025_pst_eff_a1_b0.5_phiauto"
    ]

    return clean_prs_df


def clean_pc_data(pc_data):
    """Cleans the ancestry principal components data."""
    pc_data["id_clean"] = (
        pc_data["#IID"]
        .astype(str)
        .str.strip()
        .str.replace(r"^\d+_", "", regex=True)
        .str.replace(r"^(.*?)(?:_\1)+$", r"\1", regex=True)
        .str.replace(r"(_[A-Z]+)?_T\d+(?:_\d+)?$", r"\1", regex=True)
    )

    id_dict = {
        "SCZ_SP_080": "SCZ_SP_080_SE_T1",
        "SCZ_SP_071": "SCZ_SP_071_re_T1",
        "SCZ_SP_073": "SCZ_SP_073_PR_T1",
    }
    for key, value in id_dict.items():
        if key in pc_data["id_clean"].values:
            pc_data["id_clean"] = pc_data["id_clean"].replace({key: value})

    pc_data = pc_data.set_index("id_clean")

    return pc_data


########################################################################################
# PRSCOPE DATA
########################################################################################


def clean_prscope_data(scz_prscope_data, bpd_prscope_data):
    """Cleans the PRSCOPE data. Takes the prscope_data file and returns a
    cleaned DataFrame.

    Args:
        scz_prscope_data (pd.DataFrame): The input dataframe containing PRSCOPE data.
        bpd_prscope_data (pd.DataFrame): The input dataframe containing BPD PRSCOPE
        data.

    Returns:
        pd.DataFrame: The cleaned PRSCOPE dataframe for feature size 24.
    """
    prsscope_predictions = pd.concat([scz_prscope_data, bpd_prscope_data], axis=0)
    prsscope_predictions = prsscope_predictions.set_index("ID")

    prsscope_predictions_feat24 = pd.DataFrame(
        index=prsscope_predictions.index
    ).rename_axis("gsa_id")
    prsscope_predictions_feat24["biotype1_pred"] = prsscope_predictions[
        "clust_1_feat_24"
    ]
    prsscope_predictions_feat24["biotype2_pred"] = prsscope_predictions[
        "clust_2_feat_24"
    ]
    prsscope_predictions_feat24["disease_status"] = prsscope_predictions["DX"]
    prsscope_predictions_feat24 = prsscope_predictions_feat24.query(
        "disease_status == 2"
    ).copy()
    prsscope_predictions_feat24["biotype"] = prsscope_predictions_feat24.apply(
        _assign_biotype, axis=1
    ).astype("category")

    return prsscope_predictions_feat24


def _assign_biotype(row):
    b1 = row["biotype1_pred"]
    b2 = row["biotype2_pred"]

    if b1 == 2 and b2 == 1:
        return "biotype1"
    elif b1 == 1 and b2 == 2:
        return "biotype2"
    elif b1 == 2 and b2 == 2:
        return "ambiguous"
    else:
        return "unclassified"
