import numpy as np
import pandas as pd
import pytest

from psycourse.data_analysis.univariate_analysis import (
    add_school_education,
    main_prs_columns,
    univariate_prs_regression,
    univariate_prs_regression_cov_diagnosis,
    univariate_prs_regression_cov_diagnosis_and_education,
)


@pytest.fixture
def prs_df():
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "sex": pd.Categorical(rng.choice(["F", "M"], n)),
            "diagnosis_sum": pd.Categorical(
                rng.choice(["Schizophrenia", "Bipolar-I Disorder"], n)
            ),
            "A_PRS": rng.normal(size=n),
            "B_PRS": rng.normal(size=n),
        },
        index=pd.Index([f"id{i}" for i in range(n)], name="id"),
    )
    for pc in range(1, 6):
        df[f"pc{pc}"] = rng.normal(size=n)
    df["prob_class_5"] = 0.2 * df["A_PRS"] + rng.normal(size=n)
    return df


def test_add_school_education_sets_zero_to_missing(prs_df):
    school = pd.Series([0, 1, 2, 3, 4] * 30, index=prs_df.index, dtype="Int8")
    phenotypic_data = pd.DataFrame({"school": school})

    df = add_school_education(prs_df, phenotypic_data)

    assert isinstance(df["school"].dtype, pd.CategoricalDtype)
    assert df["school"].isna().sum() == 30
    assert "school" not in prs_df.columns


def test_prs_cov_diagnosis_differs_from_primary(prs_df):
    primary, _ = univariate_prs_regression(prs_df)
    adjusted, n_subset = univariate_prs_regression_cov_diagnosis(prs_df)

    assert set(adjusted.index) == {"A_PRS", "B_PRS"}
    assert n_subset["A_PRS"] == len(prs_df)
    assert adjusted.loc["A_PRS", "coef"] != pytest.approx(primary.loc["A_PRS", "coef"])


def test_prs_cov_diagnosis_and_education_drops_missing_school(prs_df):
    school = pd.Series([0, 1, 2, 3, 4] * 30, index=prs_df.index, dtype="Int8")
    df = add_school_education(prs_df, pd.DataFrame({"school": school}))

    _, n_subset = univariate_prs_regression_cov_diagnosis_and_education(df)

    assert n_subset["A_PRS"] == len(prs_df) - 30


def test_main_prs_columns_excludes_lipid_pathway_prs(prs_df):
    df = prs_df.assign(Lipid_BD_PRS=prs_df["A_PRS"])

    results, _ = univariate_prs_regression(df)
    pathway_results, _ = univariate_prs_regression(df, prs_columns=["Lipid_BD_PRS"])

    assert main_prs_columns(df) == ["A_PRS", "B_PRS"]
    assert set(results.index) == {"A_PRS", "B_PRS"}
    assert list(pathway_results.index) == ["Lipid_BD_PRS"]
