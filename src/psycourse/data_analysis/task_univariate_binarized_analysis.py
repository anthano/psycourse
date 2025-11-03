import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_binarized_analysis import (
    univariate_lipids_ancova,
    univariate_prs_ancova,
)

UNIVARIATE_BINARIZED_ANALYSIS_DIR = (
    BLD_RESULTS / "results" / "univariate" / "binarized_analysis"
)

# ======================================================================================
# PRS
# ======================================================================================


def task_univariate_prs_ancova(
    script_path=SRC / "data_analysis" / "univariate_binarized_analysis.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=UNIVARIATE_BINARIZED_ANALYSIS_DIR / "univariate_prs_ancova_results.pkl",
):
    """Perform univariate PRS ANCOVA with binarized predicted labels (G5 vs. others)."""

    multimodal_df = pd.read_pickle(multimodal_data_path)
    prs_ancova_results = univariate_prs_ancova(multimodal_df)
    prs_ancova_results.to_pickle(produces)


# ======================================================================================
# Lipids
# ======================================================================================

lipid_ancova_products = {
    "lipid_ancova_results": UNIVARIATE_BINARIZED_ANALYSIS_DIR
    / "univariate_lipids_ancova_results.pkl",
    "lipid_ancova_top20_results": UNIVARIATE_BINARIZED_ANALYSIS_DIR
    / "univariate_lipids_ancova_top20_results.pkl",
}


def task_univariate_lipids_ancova(
    script_path=SRC / "data_analysis" / "univariate_binarized_analysis.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=UNIVARIATE_BINARIZED_ANALYSIS_DIR / "univariate_lipids_ancova_results.pkl",
):
    """Perform univariate lipids ANCOVA with binarized predicted labels."""

    multimodal_df = pd.read_pickle(multimodal_data_path)
    lipids_ancova_results, top20 = univariate_lipids_ancova(multimodal_df)
    lipids_ancova_results.to_pickle(lipid_ancova_products["lipid_ancova_results"])
    top20.to_pickle(lipid_ancova_products["lipid_ancova_top20_results"])
