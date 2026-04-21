import pickle

import pandas as pd
from pytask import task

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.descriptive_stats.descriptive_tables import (
    create_lipid_table,
    get_demographics_table,
    get_participants_per_analysis,
)

DEMOGRAPHICS_INPUT_PATH = {
    "df": BLD_DATA / "multimodal_complete_df.pkl",
    "lipid_df": BLD_DATA / "multimodal_lipid_subset_df.pkl",
    "prs_df": BLD_DATA / "multimodal_prs_subset_df.pkl",
}

DEMOGRAPHICS_OUTPUT_PATH = {
    "pkl": BLD_RESULTS / "descriptive_stats" / "demographics.pkl",
    "csv": BLD_RESULTS / "descriptive_stats" / "demographics.csv",
    "md": BLD_RESULTS / "descriptive_stats" / "demographics.md",
}


def task_get_demographics_table(
    script_path=SRC / "descriptive_stats" / "descriptive_tables.py",
    input_data_path=DEMOGRAPHICS_INPUT_PATH,
    produces=DEMOGRAPHICS_OUTPUT_PATH,
):
    df = pd.read_pickle(DEMOGRAPHICS_INPUT_PATH["df"])
    lipid_df = pd.read_pickle(DEMOGRAPHICS_INPUT_PATH["lipid_df"])
    prs_df = pd.read_pickle(DEMOGRAPHICS_INPUT_PATH["prs_df"])

    demographics_table = get_demographics_table(df, df_subset=lipid_df, df_prs=prs_df)

    demographics_table.to_pickle(DEMOGRAPHICS_OUTPUT_PATH["pkl"])
    demographics_table.to_csv(DEMOGRAPHICS_OUTPUT_PATH["csv"], index=False)
    demographics_table.to_markdown(DEMOGRAPHICS_OUTPUT_PATH["md"], index=False)


@task()
def task_create_lipid_table(
    script_path=SRC / "descriptive_stats" / "descriptive_tables.py",
    input_data_path=BLD_DATA / "cleaned_lipid_class_data.pkl",
    produces=BLD_RESULTS / "tables" / "lipid_data_table.pkl",
):
    cleaned_lipid_class_data = pd.read_pickle(input_data_path)
    lipid_data = create_lipid_table(cleaned_lipid_class_data)

    lipid_data.to_pickle(produces)
    lipid_data.to_csv(produces.with_suffix(".csv"))


########################################################################################
# Regression analysis number of participants
########################################################################################
ANALYSIS_CONFIGS = {
    "prs": {
        "standard_cov": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "prs"
        / "n_subset_dict_standard_cov.pkl",
    },
    "lipid": {
        "standard_cov": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "lipid"
        / "n_subset_dict.pkl",
        "diagnosis_cov": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "lipid"
        / "n_subset_dict_cov_diagnosis.pkl",
        "medication_cov": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "lipid"
        / "n_subset_dict_cov_med.pkl",
    },
}


def _create_task(modality):
    """Helper to create task specification for each modality."""
    config = ANALYSIS_CONFIGS[modality]

    # Determine the covariate names (exclude 'standard_cov')
    cov_names = [k for k in config.keys() if k != "standard_cov"]

    kwargs = {
        "modality": modality,
        "depends_on": list(config.values()),
        "produces": BLD_RESULTS
        / "descriptive_stats"
        / f"n_per_analysis_{modality}.pkl",
        "standard_cov_path": config["standard_cov"],
        "cov_names": cov_names,
        "cov_paths": [config[cov_name] for cov_name in cov_names],
    }

    return kwargs


# Generate tasks for each modality
for modality in ["prs", "lipid"]:

    @task(id=modality, kwargs=_create_task(modality))
    def task_get_participants_per_analysis(
        modality,
        depends_on,
        produces,
        standard_cov_path,
        cov_names,
        cov_paths,
    ):
        """Create participant count table for each analysis."""

        # Load the dictionaries
        with open(standard_cov_path, "rb") as file:
            standard_cov_dict = pickle.load(file)
        covariate_dicts = []
        for cov_path in cov_paths:
            with open(cov_path, "rb") as file:
                covariate_dicts.append(pickle.load(file))

        # Create dataframe
        n_per_analysis_df = get_participants_per_analysis(
            standard_cov_dict,
            *covariate_dicts,
            covariate_names=cov_names,
        )

        # Save output
        n_per_analysis_df.to_pickle(produces)
