import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.univariate_analysis import (
    lipid_class_prs_regression,
    lipid_prs_regression,
    prs_cv_delta_mse,
    univariate_lipid_regression,
    univariate_lipid_regression_cov_antidepressants,
    univariate_lipid_regression_cov_antipsychotics,
    univariate_lipid_regression_cov_diagnosis,
    univariate_lipid_regression_cov_med,
    univariate_lipid_regression_cov_med_and_diag,
    univariate_lipid_regression_cov_mood_stabilizers,
    univariate_lipid_regression_cov_panss,
    univariate_lipid_regression_cov_panss_gen,
    univariate_lipid_regression_cov_panss_neg,
    univariate_lipid_regression_cov_panss_pos_neg,
    univariate_lipid_regression_cov_panss_total,
    univariate_lipid_regression_cov_tranquilizers,
    univariate_prs_ancova,
    univariate_prs_regression,
)

UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "prs"
)
UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid"
)
UNIVARIATE_LIPID_MED_ADJ_RESULTS_DIR = (
    BLD_RESULTS / "univariate" / "continuous_analysis" / "lipid" / "medication_adjusted"
)

# ======================================================================================
# PRS TASKS
# ======================================================================================

univariate_prs_products = {
    "univariate_prs_results": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "univariate_prs_results_standard_cov.pkl",
    "n_subset_dict": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_standard_cov.pkl",
}


def task_univariate_prs_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_prs_products,
):
    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_results, n_subset_dict = univariate_prs_regression(data)
    univariate_prs_results.to_pickle(univariate_prs_products["univariate_prs_results"])
    pd.to_pickle(n_subset_dict, univariate_prs_products["n_subset_dict"])


# ======================================================================================

task_univariate_prs_ancova_produces = {
    "prs_extremes_ancova_results[50]": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_extremes_ancova_results_50.pkl",
    "prs_extremes_ancova_results[100]": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_extremes_ancova_results_100.pkl",
    "prs_extremes_ancova_results[120]": UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR
    / "prs_extremes_ancova_results_120.pkl",
}


def task_univariate_prs_ancova(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=task_univariate_prs_ancova_produces,
):
    """Perform ANCOVA on PRS data focussing on extreme cases."""

    data = pd.read_pickle(multimodal_df_path)
    univariate_prs_ancova_results = univariate_prs_ancova(data)
    univariate_prs_ancova_results[50].to_pickle(
        produces["prs_extremes_ancova_results[50]"]
    )
    univariate_prs_ancova_results[100].to_pickle(
        produces["prs_extremes_ancova_results[100]"]
    )
    univariate_prs_ancova_results[120].to_pickle(
        produces["prs_extremes_ancova_results[120]"]
    )


def task_prs_cv_delta_mse(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=UNIVARIATE_PRS_CONTINUOUS_RESULTS_DIR / "prs_cv_delta_mse_results.pkl",
):
    """Perform cross-validated delta MSE analysis on PRS data."""

    data = pd.read_pickle(multimodal_df_path)
    prs_cv_delta_mse_results = prs_cv_delta_mse(data)
    prs_cv_delta_mse_results.to_pickle(produces)


# ======================================================================================
# LIPID TASKS
# ======================================================================================

univariate_lipid_regression_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR / "n_subset_dict.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results.pkl",
}


def task_univariate_lipid_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = univariate_lipid_regression(
        data
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================

univariate_lipid_regression_cov_diagnosis_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_diagnosis.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_diagnosis.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_diagnosis.pkl",
}


def task_univariate_lipid_regression_cov_diagnosis(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_diagnosis_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_diagnosis(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================
univariate_lipid_regression_cov_med_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_med.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_med.pkl",
}


def task_univariate_lipid_regression_cov_med(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_med_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_med(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================
univariate_lipid_regression_cov_med_and_diag_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_med_and_diag.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_med_and_diag.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_med_and_diag.pkl",
}


def task_univariate_lipid_regression_cov_med_and_diag(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_med_and_diag_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_med_and_diag(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================
# MEDICATION-ADJUSTED (one medication class at a time)
# ======================================================================================

_MED_VARIANTS = {
    "antidepressants": univariate_lipid_regression_cov_antidepressants,
    "antipsychotics": univariate_lipid_regression_cov_antipsychotics,
    "tranquilizers": univariate_lipid_regression_cov_tranquilizers,
    "mood_stabilizers": univariate_lipid_regression_cov_mood_stabilizers,
}

for _med_name, _med_fn in _MED_VARIANTS.items():
    _produces = {
        "n_subset_dict": UNIVARIATE_LIPID_MED_ADJ_RESULTS_DIR
        / f"n_subset_dict_cov_{_med_name}.pkl",
        "top20_lipids": UNIVARIATE_LIPID_MED_ADJ_RESULTS_DIR
        / f"univariate_lipid_results_top20_cov_{_med_name}.pkl",
        "univariate_lipid_results": UNIVARIATE_LIPID_MED_ADJ_RESULTS_DIR
        / f"univariate_lipid_results_cov_{_med_name}.pkl",
    }

    def _make_task(fn, prods):
        def task_fn(
            script_path=SRC / "data_analysis" / "univariate_analysis.py",
            multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
            produces=prods,
        ):
            data = pd.read_pickle(multimodal_df_path)
            n_subset_dict, top20_lipids, univariate_lipid_results = fn(data)
            pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
            top20_lipids.to_pickle(produces["top20_lipids"])
            univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])

        return task_fn

    globals()[f"task_univariate_lipid_regression_cov_{_med_name}"] = _make_task(
        _med_fn, _produces
    )

# ======================================================================================
univariate_lipid_regression_cov_panss_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_panss.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_panss.pkl",
}


def task_univariate_lipid_regression_cov_panss(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_panss_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_panss(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================

univariate_lipid_regression_cov_panss_neg_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_panss_neg.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_neg.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_panss_neg.pkl",
}


def task_univariate_lipid_regression_cov_panss_neg(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_panss_neg_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_panss_neg(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================

univariate_lipid_regression_cov_panss_gen_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_panss_gen.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_gen.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_panss_gen.pkl",
}


def task_univariate_lipid_regression_cov_panss_gen(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_panss_gen_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_panss_gen(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================

univariate_lipid_regression_cov_panss_total_score_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_panss_total_score.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_total_score.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_panss_total_score.pkl",
}


def task_univariate_lipid_regression_cov_panss_total(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_panss_total_score_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_panss_total(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


# ======================================================================================

univariate_lipid_regression_cov_panss_both_produces = {
    "n_subset_dict": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "n_subset_dict_cov_panss_both.pkl",
    "top20_lipids": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_top20_cov_panss_both.pkl",
    "univariate_lipid_results": UNIVARIATE_LIPID_CONTINUOUS_RESULTS_DIR
    / "univariate_lipid_results_cov_panss_both.pkl",
}


def task_univariate_lipid_regression_cov_panss_both(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=univariate_lipid_regression_cov_panss_both_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset_dict, top20_lipids, univariate_lipid_results = (
        univariate_lipid_regression_cov_panss_pos_neg(data)
    )
    pd.to_pickle(n_subset_dict, produces["n_subset_dict"])
    top20_lipids.to_pickle(produces["top20_lipids"])
    univariate_lipid_results.to_pickle(produces["univariate_lipid_results"])


########################################################################################
# Lipid X PRS Regression
########################################################################################

prs_types = ["BD", "MDD", "SCZ", "Lipid_BD", "Lipid_MDD", "Lipid_SCZ"]
lipid_prs_result_dir = BLD_RESULTS / "univariate" / "continuous_analysis" / "lipidXprs"


lipid_prs_regression_produces = (
    {
        f"{prs}_Lipid_association_top20": lipid_prs_result_dir
        / f"{prs}_Lipid_association_top20.pkl"
        for prs in prs_types
    }
    | {
        f"{prs}_Lipid_association": lipid_prs_result_dir
        / f"{prs}_Lipid_association.pkl"
        for prs in prs_types
    }
    | {"n_subset": lipid_prs_result_dir / "PRS_Lipid_association_n_subset.pkl"}
)


def task_lipid_prs_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=lipid_prs_regression_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset, top_by_prs, results_by_prs = lipid_prs_regression(data)

    for prs in prs_types:
        top_by_prs[f"{prs}_PRS"].to_pickle(produces[f"{prs}_Lipid_association_top20"])
        results_by_prs[f"{prs}_PRS"].to_pickle(produces[f"{prs}_Lipid_association"])

    pd.to_pickle(n_subset, produces["n_subset"])


# ======================================================================================

lipid_class_prs_regression_produces = (
    {
        f"{prs}_Lipid_class_association_top20": lipid_prs_result_dir
        / f"{prs}_Lipid_class_association_top20.pkl"
        for prs in prs_types
    }
    | {
        f"{prs}_Lipid_class_association": lipid_prs_result_dir
        / f"{prs}_Lipid_class_association.pkl"
        for prs in prs_types
    }
    | {"n_subset": lipid_prs_result_dir / "PRS_Lipid_class_association_n_subset.pkl"}
)


def task_lipid_class_prs_regression(
    script_path=SRC / "data_analysis" / "univariate_analysis.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=lipid_class_prs_regression_produces,
):
    data = pd.read_pickle(multimodal_df_path)
    n_subset, top_by_prs, results_by_prs = lipid_class_prs_regression(data)

    for prs in prs_types:
        top_by_prs[f"{prs}_PRS"].to_pickle(
            produces[f"{prs}_Lipid_class_association_top20"]
        )
        results_by_prs[f"{prs}_PRS"].to_pickle(
            produces[f"{prs}_Lipid_class_association"]
        )

    pd.to_pickle(n_subset, produces["n_subset"])
