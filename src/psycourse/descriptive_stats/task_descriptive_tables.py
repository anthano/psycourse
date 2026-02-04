import pickle

import pandas as pd
from pytask import task

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.descriptive_stats.descriptive_tables import (
    create_lipid_table,
    get_participants_per_analysis,
)


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
        "diagnosis_cov": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "prs"
        / "n_subset_dict_cov_diagnosis.pkl",
        "bmi_cov": BLD_RESULTS
        / "univariate"
        / "continuous_analysis"
        / "prs"
        / "n_subset_dict_cov_bmi.pkl",
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
        / f"n_per_analysis_{modality}.csv",
        "cov_1_name": cov_names[0],
        "cov_2_name": cov_names[1],
        "standard_cov_path": config["standard_cov"],
        "cov_1_path": config[cov_names[0]],
        "cov_2_path": config[cov_names[1]],
    }

    return kwargs


# Generate tasks for each modality
for modality in ["prs", "lipid"]:

    @task(id=modality, kwargs=_create_task(modality))
    def task_get_participants_per_analysis(
        modality,
        depends_on,
        produces,
        cov_1_name,
        cov_2_name,
        standard_cov_path,
        cov_1_path,
        cov_2_path,
    ):
        """Create participant count table for each analysis."""

        # Load the dictionaries
        with open(standard_cov_path, "rb") as file:
            standard_cov_dict = pickle.load(file)

        with open(cov_1_path, "rb") as file:
            cov_1_dict = pickle.load(file)

        with open(cov_2_path, "rb") as file:
            cov_2_dict = pickle.load(file)

        # Create dataframe
        n_per_analysis_df = get_participants_per_analysis(
            standard_cov_dict,
            cov_1_dict,
            cov_2_dict,
            cov_1_name=cov_1_name,
            cov_2_name=cov_2_name,
        )

        # Save output
        n_per_analysis_df.to_csv(produces)
