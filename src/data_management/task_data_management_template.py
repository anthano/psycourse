"""Tasks for managing the data."""

import pandas as pd
from psycourse.config import BLD, SRC

from data_management.data_cleaning.py import clean_phenotypic_data


def task_clean_phenotypic_data(
    script=SRC / "data_management" / "data_cleaning.py",
    data=SRC / "data" / "230614_v6.0" / "230614_v6.0_psycourse_wd.csv",
    produces=BLD / "data" / "cleaned_phenotypic_data.pkl",
):
    """Clean the stats4schools smoking data set."""
    data = pd.read_csv(data, delimiter="\t")
    clean_data = clean_phenotypic_data(data)
    clean_data.to_pickle(produces)
