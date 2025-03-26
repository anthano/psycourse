import numpy as np
import pandas as pd

from psycourse.ml_pipeline.train_model import (
    CovariateResidualizer,
    DataFrameImputer,
)

# def test_identify_continuous_columns():
#    # Create a simple DataFrame with different types of columns.
#    df = pd.DataFrame({
#        'A': [1, 2, 3],
#        'B': [4.0, 5.0, 6.0],
#        'C': ['a', 'b', 'c']
#    })
#    # Identify continuous columns
#    continuous_cols = _identify_continuous_cols(df)
#    # Check that the continuous columns are correctly identified.
#    assert continuous_cols == ['A', 'B']


def test_dataframe_imputer():
    # Create a simple DataFrame with missing values.
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})

    # Use a simple imputer, e.g., fill NaN with median
    class DummyImputer:
        def fit(self, X):
            return self

        def transform(self, X):
            return X.fillna(X.median())

    imputer = DataFrameImputer(DummyImputer())
    imputer.fit(df)
    df_imputed = imputer.transform(df)
    # Check that there are no NaNs
    assert df_imputed.isna().sum().sum() == 0


def test_covariate_residualizer():
    # Create a simple DataFrame where one column is a linear combination of covariates.
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "bmi": [22, 24, 26],
            "sex": [0, 1, 0],
            "feature": [50, 70, 90],
        }
    )
    covariates = ["age", "bmi", "sex"]
    # In this example, assume feature = 2*age + 1*bmi + 1*sex
    residualizer = CovariateResidualizer(covariate_cols=covariates)
    residualizer.fit(df, None)
    df_resid = residualizer.transform(df)
    # After residualization, feature should be near zero.
    np.testing.assert_allclose(
        df_resid["feature"], [0, 0, 0], atol=1e-5
    )  # adjust expected values
