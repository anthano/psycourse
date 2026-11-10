import numpy as np
import pandas as pd
import pytest

from psycourse.data_analysis.univariate_analysis import (
    main_prs_columns,
    univariate_prs_regression,
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


def test_main_prs_columns_excludes_lipid_pathway_prs(prs_df):
    df = prs_df.assign(Lipid_BD_PRS=prs_df["A_PRS"])

    results, _ = univariate_prs_regression(df)
    pathway_results, _ = univariate_prs_regression(df, prs_columns=["Lipid_BD_PRS"])

    assert main_prs_columns(df) == ["A_PRS", "B_PRS"]
    assert set(results.index) == {"A_PRS", "B_PRS"}
    assert list(pathway_results.index) == ["Lipid_BD_PRS"]
