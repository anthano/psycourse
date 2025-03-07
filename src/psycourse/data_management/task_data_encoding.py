import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.data_management.data_encoding import encode_and_prune_data


def task_encode_and_prune_data(
    script_path=SRC / "data_management" / "data_encoding.py",
    data_path=BLD_DATA / "cleaned_phenotypic_data.pkl",
    produces=BLD_DATA / "encoded_phenotypic_data.pkl",
):
    data = pd.read_pickle(data_path)
    encoded_df = encode_and_prune_data(data)
    encoded_df.to_pickle(produces)
