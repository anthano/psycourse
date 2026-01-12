import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_prep_multimodal_lipid_subset import (
    subset_multimodal_data_for_lipids,
)


def task_data_prep_multimodal_lipid_subset(
    script_path=SRC / "data_management" / "data_prep_multimodal_lipid_subset.py",
    data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_DATA / "multimodal_lipid_subset_df.pkl",
):
    data = pd.read_pickle(data_path)
    lipid_subset_multimodal_df = subset_multimodal_data_for_lipids(data)
    lipid_subset_multimodal_df.to_pickle(produces)
