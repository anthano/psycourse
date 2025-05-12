import pandas as pd


def concatenate_features_and_targets(phenotypic_df, target_df):
    """Concatenate the features and targets into a single dataframe." """

    return pd.concat([phenotypic_df, target_df], axis=1, join="inner")


def create_sparse_dataset_for_classifier(phenotypic_df, target_df):
    """Create a sparse dataset for the classifier."""

    sparse_phenotypic_df = pd.DataFrame(index=phenotypic_df.index)
    sparse_phenotypic_df["partner"] = phenotypic_df["partner"]
    sparse_phenotypic_df["whoqol_13"] = phenotypic_df["whoqol_13"]
    sparse_phenotypic_df["whoqol_23"] = phenotypic_df["whoqol_23"]
    sparse_phenotypic_df["whoqol_14"] = phenotypic_df["whoqol_14"]
    sparse_phenotypic_df["outpat_treatment"] = phenotypic_df["outpat_treatment"]
    sparse_phenotypic_df["whoqol_25"] = phenotypic_df["whoqol_25"]
    sparse_phenotypic_df["whoqol_6"] = phenotypic_df["whoqol_6"]
    sparse_phenotypic_df["whoqol_12"] = phenotypic_df["whoqol_12"]
    sparse_phenotypic_df["whoqol_env"] = phenotypic_df["whoqol_env"]
    sparse_phenotypic_df["whoqol_20"] = phenotypic_df["whoqol_20"]
    sparse_phenotypic_df["suic_attempt"] = phenotypic_df[
        "suic_attempt"
    ]  # note: encoded differently
    sparse_phenotypic_df["severity_suic_ide"] = phenotypic_df[
        "severity_suic_ide"
    ]  # note: encoded differently
    sparse_phenotypic_df["suic_methods"] = phenotypic_df[
        "suic_methods"
    ]  # note: encoded differently
    sparse_phenotypic_df["prep_suic_attempt_ord"] = phenotypic_df[
        "prep_suic_attempt_ord"
    ]  # note: encoded differently
    sparse_phenotypic_df["no_suic_attempt"] = phenotypic_df[
        "no_suic_attempt"
    ]  # note: encoded differently
    sparse_phenotypic_df["ever_suic_ide"] = phenotypic_df["ever_suic_ide"]
    sparse_phenotypic_df["suic_note"] = phenotypic_df[
        "suic_note"
    ]  # note: encoded differently
    sparse_phenotypic_df["suic_attempt_note"] = phenotypic_df[
        "suic_attempt_note"
    ]  # note: encoded differently
    sparse_phenotypic_df["fam_hist"] = phenotypic_df["fam_hist"]
    sparse_phenotypic_df["med_change"] = phenotypic_df["med_change"]
    sparse_phenotypic_df["work_impairment"] = phenotypic_df["work_impairment"]
    sparse_phenotypic_df["idsc_5"] = phenotypic_df["idsc_5"]
    sparse_phenotypic_df["idsc_16"] = phenotypic_df["idsc_16"]
    sparse_phenotypic_df["idsc_total"] = phenotypic_df["idsc_total"]
    sparse_phenotypic_df["inf_disease"] = phenotypic_df["inf_disease"]
    sparse_phenotypic_df["ever_delus"] = phenotypic_df["ever_delus"]
    sparse_phenotypic_df["idsc_15"] = phenotypic_df["idsc_15"]
    sparse_phenotypic_df["adverse_events_curr_medication"] = phenotypic_df[
        "adverse_events_curr_medication"
    ]
    sparse_phenotypic_df["marital_stat_Single"] = phenotypic_df["marital_stat_Single"]
    sparse_phenotypic_df["illicit_drugs"] = phenotypic_df["illicit_drugs"]
    sparse_phenotypic_df["whoqol_3"] = phenotypic_df["whoqol_3"]
    sparse_phenotypic_df["smoker_yes"] = phenotypic_df["smoker_yes"]
    sparse_phenotypic_df["sex"] = phenotypic_df["sex"]
    sparse_phenotypic_df["language_skill_mother tongue"] = phenotypic_df[
        "language_skill_mother tongue"
    ]
    sparse_phenotypic_df["school"] = phenotypic_df["school"]
    sparse_phenotypic_df["whoqol_15"] = phenotypic_df["whoqol_15"]
    sparse_phenotypic_df["current_psych_treatment"] = phenotypic_df[
        "current_psych_treatment"
    ]
    sparse_phenotypic_df["ever_halluc"] = phenotypic_df["ever_halluc"]
    sparse_phenotypic_df["panss_p1"] = phenotypic_df["panss_p1"]
    sparse_phenotypic_df["panss_total_score"] = phenotypic_df["panss_total_score"]
    sparse_phenotypic_df["panss_sum_pos"] = phenotypic_df["panss_sum_pos"]
    sparse_phenotypic_df["panss_p2"] = phenotypic_df["panss_p2"]
    sparse_phenotypic_df["panss_sum_neg"] = phenotypic_df["panss_sum_neg"]
    sparse_phenotypic_df["panss_n1"] = phenotypic_df["panss_n1"]
    sparse_phenotypic_df["panss_sum_gen"] = phenotypic_df["panss_sum_gen"]

    sparse_phenotypic_df_with_targets = pd.concat(
        [sparse_phenotypic_df, target_df], axis=1, join="inner"
    )

    sparse_phenotypic_new_df_without_targets = sparse_phenotypic_df.loc[
        ~sparse_phenotypic_df.index.isin(target_df.index)
    ]

    return sparse_phenotypic_df_with_targets, sparse_phenotypic_new_df_without_targets
