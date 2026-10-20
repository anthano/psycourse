import numpy as np
import pandas as pd
import pytest

from psycourse.data_analysis.incremental_r2 import (
    R2_COMPONENTS,
    bootstrap_incremental_r2,
    incremental_r2_decomposition,
    significant_predictors,
)

PRS_COLS = ["A_PRS", "B_PRS", "C_PRS"]
LIPID_COLS = ["PC", "PE", "dSM", "TAG"]


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "sex": pd.Categorical(rng.choice(["F", "M"], n)),
            "bmi": rng.normal(28, 5, n),
            "duration_illness": rng.normal(12, 8, n),
            "smoker": pd.Categorical(rng.choice(["never", "former", "yes"], n)),
        }
    )
    for pc in range(1, 6):
        df[f"pc{pc}"] = rng.normal(size=n)
    for col in PRS_COLS + LIPID_COLS:
        df[col] = rng.normal(size=n)
    df["prob_class_5"] = 0.3 * df["A_PRS"] + 0.3 * df["PC"] + rng.normal(size=n)
    df.loc[:4, "A_PRS"] = np.nan  # exercises median imputation
    return df


def test_bootstrap_point_estimates_match_decomposition(synthetic_df):
    results = incremental_r2_decomposition(
        synthetic_df, PRS_COLS, LIPID_COLS, n_permutations=10
    )
    table = bootstrap_incremental_r2(
        synthetic_df, PRS_COLS, LIPID_COLS, n_bootstraps=50
    )

    for key, label in R2_COMPONENTS.items():
        if key == "shared":
            continue
        assert table.loc[label, ("Residualized", "ΔR²")] == pytest.approx(
            results[key], abs=1e-6
        )
        assert table.loc[label, ("Traditional (raw)", "ΔR²")] == pytest.approx(
            results[f"{key}_trad"], abs=1e-6
        )


def test_bootstrap_shared_component_and_ci_order(synthetic_df):
    table = bootstrap_incremental_r2(
        synthetic_df, PRS_COLS, LIPID_COLS, n_bootstraps=50
    )
    resid = table["Residualized"]

    expected_shared = (
        resid.loc["PRS alone", "ΔR²"]
        + resid.loc["Lipids alone", "ΔR²"]
        - resid.loc["PRS + Lipids jointly", "ΔR²"]
    )
    assert resid.loc["Shared (PRS ∩ Lipids)", "ΔR²"] == pytest.approx(
        expected_shared, abs=1e-5
    )
    assert (resid["95% CI lower"] <= resid["95% CI upper"]).all()
    assert table.attrs["n"] == len(synthetic_df)


def test_significant_predictors_uses_fdr_threshold():
    results = pd.DataFrame(
        {"FDR": [0.001, 0.049, 0.05, 0.3]}, index=["PC", "dSM", "PE", "LPE"]
    )

    assert significant_predictors(results) == ["PC", "dSM"]
