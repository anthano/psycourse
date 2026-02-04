import pandas as pd
from pytask import task

from psycourse.config import BLD_RESULTS, WRITING


@task(
    kwargs={
        "depends_on": {
            "lipid_list": BLD_RESULTS / "tables" / "lipid_data_table.pkl",
            "n_per_regression_analysis_lipid": BLD_RESULTS
            / "descriptive_stats"
            / "n_per_analysis_lipid.pkl",
            "n_per_regression_analysis_prs": BLD_RESULTS
            / "descriptive_stats"
            / "n_per_analysis_prs.pkl",
            "prs_regression_result": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "prs"
            / "univariate_prs_results_standard_cov.pkl",
            "prs_regression_result_cov_bmi": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "prs"
            / "univariate_prs_results_cov_bmi.pkl",
            "prs_regression_result_cov_diagnosis": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "prs"
            / "univariate_prs_results_cov_diagnosis.pkl",
            "lipid_regression_result": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "lipid"
            / "univariate_lipid_results.pkl",
            "lipid_regression_result_cov_diagnosis": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "lipid"
            / "univariate_lipid_results_cov_diagnosis.pkl",
            "lipid_regression_result_cov_medication": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "lipid"
            / "univariate_lipid_results_cov_med.pkl",
            "lipid_enrichment_result": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "lipid"
            / "lipid_enrichment_results.pkl",
            "lipid_enrichment_result_cov_diagnosis": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "lipid"
            / "lipid_enrichment_results_cov_diagnosis.pkl",
            "lipid_enrichment_result_cov_medication": BLD_RESULTS
            / "univariate"
            / "continuous_analysis"
            / "lipid"
            / "lipid_enrichment_results_cov_med.pkl",
            "mediation_analysis_result": BLD_RESULTS
            / "mediation_analysis"
            / "mediation_analysis_results.pkl",
            # ... add other tables here
        },
        "produces": WRITING / "supplementary_tables.xlsx",
    }
)
def task_export_all_supplementary_tables(depends_on, produces):
    with pd.ExcelWriter(produces, engine="openpyxl") as writer:
        for i, (name, path) in enumerate(depends_on.items(), start=1):
            df = pd.read_pickle(path)
            # Sheet names limited to 31 chars in Excel
            sheet_name = f"{i:02d}_{name}"[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=True)
