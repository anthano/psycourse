import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.ml_pipeline.concat_full_predicted_probability_df import (
    concat_full_predicted_probability_df,
)
from psycourse.ml_pipeline.out_of_fold_predictions import (
    svm_out_of_fold_probabilities,
)

oof_products = {
    "oof_probabilities": BLD_DATA / "svm_oof_probabilities.pkl",
    "oof_probabilities_full": BLD_DATA / "svm_oof_probabilities_full.pkl",
    "fold_summary": BLD_RESULTS / "svm" / "oof_fold_summary.pkl",
}


def task_svm_out_of_fold_probabilities(
    script_path=SRC / "ml_pipeline" / "out_of_fold_predictions.py",
    data_path=BLD_DATA / "sparse_dataset_with_targets.pkl",
    new_probabilities_path=BLD_DATA
    / "svm_predicted_probabilities_new_psycourse_data.pkl",
    produces=oof_products,
):
    """
    Out-of-fold probabilities for the participants with cluster labels, combined
    with the (already out-of-sample) probabilities of the new PsyCourse participants.
    """
    data = pd.read_pickle(data_path)
    new_probabilities = pd.read_pickle(new_probabilities_path)

    oof_probabilities, fold_summary = svm_out_of_fold_probabilities(data)
    oof_probabilities_full = concat_full_predicted_probability_df(
        oof_probabilities.drop(columns="outer_fold"), new_probabilities
    )

    oof_probabilities.to_pickle(produces["oof_probabilities"])
    oof_probabilities_full.to_pickle(produces["oof_probabilities_full"])
    fold_summary.to_pickle(produces["fold_summary"])
