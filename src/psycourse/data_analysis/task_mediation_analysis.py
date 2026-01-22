from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.mediation_analysis import mediation_analysis


def task_mediation_analysis(
    script_path: Path = SRC / "data_analysis" / "mediation_analysis.py",
    multimodal_lipid_subset_df_path: Path = BLD_DATA / "multimodal_lipid_subset_df.pkl",
    lipid_enrichment_result_df_path: Path = BLD_RESULTS
    / "univariate"
    / "continuous_analysis"
    / "lipid"
    / "lipid_enrichment_results.pkl",
    produces: Annotated[Path, Product] = BLD_RESULTS
    / "mediation_analysis"
    / "mediation_analysis_results.pkl",
):
    df = pd.read_pickle(multimodal_lipid_subset_df_path)
    lipid_enrichment_result_df = pd.read_pickle(lipid_enrichment_result_df_path)
    mediation_results = mediation_analysis(df, lipid_enrichment_result_df)
    mediation_results.to_pickle(produces)
