import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.lipid_permutation_analysis import (
    lipid_class_enrichment_gsea,
)


def task_univariate_prs_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    regression_results_df_path=BLD_RESULTS
    / "univariate"
    / "continuous_analysis"
    / "lipid"
    / "univariate_lipid_results.pkl",
    annot_df_path=BLD_DATA / "cleaned_lipid_class_data.pkl",
    produces=BLD_RESULTS
    / "univariate"
    / "continuous_analysis"
    / "lipid"
    / "lipid_enrichment_results.pkl",
):
    results_df = pd.read_pickle(regression_results_df_path)
    annot_df = pd.read_pickle(annot_df_path)

    enrichment_results = lipid_class_enrichment_gsea(results_df, annot_df)
    enrichment_results.to_pickle(produces)
