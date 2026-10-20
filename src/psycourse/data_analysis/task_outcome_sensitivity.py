import pickle

import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.data_analysis.incremental_r2 import incremental_r2_decomposition
from psycourse.data_analysis.lipid_permutation_analysis import (
    lipid_class_enrichment_gsea,
)
from psycourse.data_analysis.outcome_sensitivity import (
    compare_outcomes,
    compare_r2,
    compare_results,
    with_outcome,
)
from psycourse.data_analysis.task_incremental_r2 import lipid_cols, prs_cols
from psycourse.data_analysis.univariate_analysis import (
    univariate_lipid_regression,
    univariate_lipid_regression_cov_diagnosis,
    univariate_lipid_regression_cov_med,
    univariate_lipid_regression_cov_med_and_diag,
    univariate_prs_regression,
    univariate_prs_regression_cov_diagnosis,
)

# Sensitivity analysis: key analyses re-run with the in-sample SVM probabilities
# (the primary outcome is the out-of-fold probability).
RESULTS_DIR = BLD_RESULTS / "outcome_sensitivity_in_sample"
IN_SAMPLE_PROBABILITIES_PATH = BLD_DATA / "svm_predicted_probabilities_full.pkl"
_UNI = BLD_RESULTS / "univariate" / "continuous_analysis"

# ======================================================================================
# PRS
# ======================================================================================

in_sample_prs_products = {
    "prs_standard_cov": RESULTS_DIR / "univariate_prs_results_standard_cov.pkl",
    "prs_cov_diagnosis": RESULTS_DIR / "univariate_prs_results_cov_diagnosis.pkl",
}


def task_in_sample_prs_regression(
    script_path=SRC / "data_analysis" / "outcome_sensitivity.py",
    multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
    probabilities_path=IN_SAMPLE_PROBABILITIES_PATH,
    produces=in_sample_prs_products,
):
    """Primary and diagnosis-adjusted PRS models with the in-sample outcome."""
    data = with_outcome(
        pd.read_pickle(multimodal_df_path), pd.read_pickle(probabilities_path)
    )
    univariate_prs_regression(data)[0].to_pickle(produces["prs_standard_cov"])
    univariate_prs_regression_cov_diagnosis(data)[0].to_pickle(
        produces["prs_cov_diagnosis"]
    )


# ======================================================================================
# LIPID SPECIES + CLASS ENRICHMENT
# ======================================================================================

_LIPID_VARIANTS = {
    "standard": univariate_lipid_regression,
    "cov_diagnosis": univariate_lipid_regression_cov_diagnosis,
    "cov_med": univariate_lipid_regression_cov_med,
    "cov_med_and_diag": univariate_lipid_regression_cov_med_and_diag,
}

for _variant, _fn in _LIPID_VARIANTS.items():
    _produces = {
        "lipid_results": RESULTS_DIR / f"univariate_lipid_results_{_variant}.pkl",
        "enrichment_results": RESULTS_DIR / f"lipid_enrichment_results_{_variant}.pkl",
    }

    def _make_lipid_task(fn, prods):
        def task_fn(
            script_path=SRC / "data_analysis" / "outcome_sensitivity.py",
            multimodal_df_path=BLD_DATA / "multimodal_complete_df.pkl",
            probabilities_path=IN_SAMPLE_PROBABILITIES_PATH,
            annot_df_path=BLD_DATA / "cleaned_lipid_class_data.pkl",
            produces=prods,
        ):
            data = with_outcome(
                pd.read_pickle(multimodal_df_path),
                pd.read_pickle(probabilities_path),
            )
            _, _, lipid_results = fn(data)
            enrichment_results = lipid_class_enrichment_gsea(
                lipid_results, pd.read_pickle(annot_df_path)
            )
            lipid_results.to_pickle(produces["lipid_results"])
            enrichment_results.to_pickle(produces["enrichment_results"])

        return task_fn

    globals()[f"task_in_sample_lipid_regression_{_variant}"] = _make_lipid_task(
        _fn, _produces
    )

# ======================================================================================
# INCREMENTAL R²
# ======================================================================================


def task_in_sample_incremental_r2(
    script_path=SRC / "data_analysis" / "outcome_sensitivity.py",
    data_path=BLD_DATA / "multimodal_lipid_subset_df.pkl",
    probabilities_path=IN_SAMPLE_PROBABILITIES_PATH,
    produces=RESULTS_DIR / "incremental_r2_results.pkl",
):
    """Incremental R² decomposition with the in-sample outcome."""
    data = with_outcome(pd.read_pickle(data_path), pd.read_pickle(probabilities_path))
    results = incremental_r2_decomposition(
        df=data,
        prs_cols=prs_cols,
        lipid_cols=lipid_cols,
        outcome_col="prob_class_5",
        n_permutations=20_000,
        random_state=42,
    )
    with open(produces, "wb") as f:
        pickle.dump(results, f)


# ======================================================================================
# SUMMARY: out-of-fold (primary) vs in-sample
# ======================================================================================

_LIPID_PRIMARY_SUFFIX = {
    "standard": "",
    "cov_diagnosis": "_cov_diagnosis",
    "cov_med": "_cov_med",
    "cov_med_and_diag": "_cov_med_and_diag",
}

summary_depends_on = {
    "multimodal_df": BLD_DATA / "multimodal_complete_df.pkl",
    "lipid_subset_df": BLD_DATA / "multimodal_lipid_subset_df.pkl",
    "in_sample_probabilities": IN_SAMPLE_PROBABILITIES_PATH,
    "annot_df": BLD_DATA / "cleaned_lipid_class_data.pkl",
    "r2_primary": BLD_RESULTS / "incremental_r2" / "incremental_r2_results.pkl",
    "r2_in_sample": RESULTS_DIR / "incremental_r2_results.pkl",
    "prs_standard_cov_primary": _UNI
    / "prs"
    / "univariate_prs_results_standard_cov.pkl",
    "prs_cov_diagnosis_primary": _UNI
    / "prs"
    / "univariate_prs_results_cov_diagnosis.pkl",
    **{f"{key}_in_sample": path for key, path in in_sample_prs_products.items()},
}
for _variant, _suffix in _LIPID_PRIMARY_SUFFIX.items():
    summary_depends_on[f"lipid_{_variant}_primary"] = (
        _UNI / "lipid" / f"univariate_lipid_results{_suffix}.pkl"
    )
    summary_depends_on[f"enrichment_{_variant}_primary"] = (
        _UNI / "lipid" / f"lipid_enrichment_results{_suffix}.pkl"
    )
    summary_depends_on[f"lipid_{_variant}_in_sample"] = (
        RESULTS_DIR / f"univariate_lipid_results_{_variant}.pkl"
    )
    summary_depends_on[f"enrichment_{_variant}_in_sample"] = (
        RESULTS_DIR / f"lipid_enrichment_results_{_variant}.pkl"
    )


def task_outcome_sensitivity_summary(
    script_path=SRC / "data_analysis" / "outcome_sensitivity.py",
    depends_on=summary_depends_on,
    produces=RESULTS_DIR / "oof_vs_in_sample_summary.xlsx",
):
    """Side-by-side comparison of key results under both outcome definitions."""
    d = {key: pd.read_pickle(path) for key, path in depends_on.items()}
    species = d["annot_df"]["lipid_species"]

    with pd.ExcelWriter(produces, engine="openpyxl") as writer:
        compare_outcomes(
            {
                "PRS sample": d["multimodal_df"].loc[
                    d["multimodal_df"]["SCZ_PRS"].notna()
                ],
                "lipid subset": d["lipid_subset_df"],
            },
            d["in_sample_probabilities"],
        ).to_excel(writer, sheet_name="outcome_agreement")

        for model in ["standard_cov", "cov_diagnosis"]:
            compare_results(
                d[f"prs_{model}_primary"], d[f"prs_{model}_in_sample"]
            ).to_excel(writer, sheet_name=f"prs_{model}")

        for variant in _LIPID_PRIMARY_SUFFIX:
            lipid_table = compare_results(
                d[f"lipid_{variant}_primary"], d[f"lipid_{variant}_in_sample"]
            )
            lipid_table.insert(0, "lipid_species", species.reindex(lipid_table.index))
            lipid_table.to_excel(writer, sheet_name=f"lipid_{variant}")
            compare_results(
                d[f"enrichment_{variant}_primary"],
                d[f"enrichment_{variant}_in_sample"],
                columns=("NES", "FDR"),
            ).to_excel(writer, sheet_name=f"enrichment_{variant}")

        compare_r2(d["r2_primary"], d["r2_in_sample"]).to_excel(
            writer, sheet_name="incremental_r2"
        )
