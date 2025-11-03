import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_binarized_analysis import univariate_prs_ancova

UNIVARIATE_BINARIZED_ANALYSIS_DIR = (
    BLD_RESULTS / "results" / "univariate" / "binarized_analysis"
)


def task_univariate_prs_ancova(
    script_path=SRC / "data_analysis" / "univariate_binarized_analysis.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=UNIVARIATE_BINARIZED_ANALYSIS_DIR / "univariate_prs_ancova_results.pkl",
):
    """Perform univariate PRS ANCOVA with binarized predicted labels (G5 vs. others)."""
    multimodal_df = pd.read_pickle(multimodal_data_path)
    prs_ancova_results = univariate_prs_ancova(multimodal_df)
    prs_ancova_results.to_pickle(produces)
