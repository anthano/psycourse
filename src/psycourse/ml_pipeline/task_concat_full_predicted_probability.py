import pandas as pd

from psycourse.config import BLD_DATA, SRC
from psycourse.ml_pipeline.concat_full_predicted_probability_df import (
    concat_full_predicted_probability_df,
)

data_path_dict = {
    "old_probabilities": BLD_DATA / "svm_predicted_probabilities.pkl",
    "new_probabilites": BLD_DATA / "svm_predicted_probabilities_new_psycourse_data.pkl",
}


def task_concat_full_predicted_probability(
    script_path=SRC / "ml_pipeline" / "concat_full_predicted_probability_df.py",
    data_path=data_path_dict,
    produces=BLD_DATA / "svm_predicted_probabilities_full.pkl",
):
    old_probabilities = pd.read_pickle(data_path["old_probabilities"])
    new_probabilities = pd.read_pickle(data_path["new_probabilites"])
    concatenated_df = concat_full_predicted_probability_df(
        old_probabilities, new_probabilities
    )
    concatenated_df.to_pickle(produces)
