import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC

from psycourse.config import BLD_DATA
from psycourse.ml_pipeline.impute import KNNMedianImputer

###############################################################################
# Main SVM Model Function
###############################################################################


def svm_model(phenotypic_df, target_df, covariate_cols):
    """
    Train an SVM model with nested cross-validation on the given data.

    Args:
        phenotypic_df (pd.DataFrame): Predictor variables (including covariates).
        target_df (pd.DataFrame): Target variable (Cluster Labels).
        covariate_cols (list): List of covariate column names (e.g., ['bmi', 'age',
            'sex']).

    Returns:
        tuple: Mean and standard deviation of nested CV scores.
    """
    phenotypic_df, target_df = _keep_only_rows_with_target(phenotypic_df, target_df)

    # Use the DataFrame directly so that column names are preserved.
    X_val = phenotypic_df.copy()
    y_val = target_df.copy()

    # Identify columns to be residualized and scaled (using column names)
    scaling_cols_names = _identify_continuous_cols(phenotypic_df, covariate_cols)

    # Define a transformer to drop the covariate columns.
    drop_covariates_transformer = FunctionTransformer(
        lambda X: X.drop(columns=covariate_cols)
    )

    # Define SVM model:
    base_svm = LinearSVC(penalty="l2", class_weight="balanced")
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=5)

    # Build the pipeline:
    pipeline = Pipeline(
        [
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=7))),
            ("residualize", CovariateResidualizer(covariate_cols=covariate_cols)),
            ("drop_covariates", drop_covariates_transformer),
            (
                "first_scaling",
                ColumnTransformer(
                    transformers=[
                        (
                            "scaler",
                            MinMaxScaler(feature_range=(-1, 1)),
                            scaling_cols_names,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("pca", PCA()),
            # Add a scaling step here again
            ("svm", OneVsRestClassifier(calibrated_svm)),
        ]
    )

    # Define grid search parameters.
    parameters_grid = {
        # "pca__n_components": [0.25, 0.5, 0.75, 1],
        "pca__n_components": [0.25],
        "svm__estimator__estimator__C": [0.015625, 0.03125],
        # "svm__estimator__estimator__C": [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]  # noqa: E501
    }

    # Set up nested cross-validation (10 inner folds and 10 outer folds)
    inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=2, shuffle=True, random_state=42)

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    nested_scores = cross_val_score(clf, X=X_val, y=y_val, cv=outer_cv)

    mean_score = nested_scores.mean()
    std_score = nested_scores.std()

    print(f"Mean Nested CV Score: {mean_score:.4f}")
    print(f"Standard Deviation of Nested CV Scores: {std_score:.4f}")

    # get probabilities for cluster 5
    X_new = phenotypic_df.copy()
    final_model = pipeline.fit(X_val, y_val)
    probabilities = final_model.predict_proba(X_new)
    print(probabilities)

    # Find the index of cluster 5 in the classes array:
    class_index = np.where(final_model.classes_ == 5)[0][0]

    # Extract the probabilities corresponding to cluster 5:
    cluster5_probs = probabilities[:, class_index]

    # Create a DataFrame using the sample IDs from X_new's index:
    result_df = pd.DataFrame(
        {"sample_id": X_new.index, "cluster5_probability": cluster5_probs}
    )

    print(result_df.head())
    return mean_score, std_score, result_df


###############################################################################
# Helper Transformers & Functions
###############################################################################


def _keep_only_rows_with_target(phenotypic_df, target_df):
    """Keep only the rows in the dataframe that have a corresponding target value.

    Args:
        phenotypic_df (pd.DataFrame): The predictor variables.
        target_df (pd.DataFrame): The target variable.

    Returns:
        pd.DataFrame: The filtered predictor variables.
    """
    # target_df = target_df.set_index("cases")

    return phenotypic_df[phenotypic_df.index.isin(target_df.index)], target_df[
        target_df.index.isin(phenotypic_df.index)
    ]


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    A wrapper for an imputer (e.g. KNNImputer) that ensures the output remains
    a DataFrame with the same columns and index as the input.
    """

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_t = self.imputer.transform(X)
        imputed = pd.DataFrame(X_t, columns=X.columns, index=X.index)
        # convert imputed integer values to integers
        # for all columns that have integer dtype (in X), floor the imputed values
        int_dtypes = [pd.Int8Dtype(), pd.Int16Dtype(), pd.Int32Dtype(), pd.Int64Dtype()]
        for col in X.select_dtypes(include=int_dtypes).columns:
            imputed[col] = imputed[col].round()
        return imputed.astype(X.dtypes)


class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """
    Transformer to residualize features by removing the linear effects of specified
    covariates.
    """

    def __init__(self, covariate_cols):
        self.covariate_cols = covariate_cols

    def fit(self, X, y=None):
        self.models_ = {}
        for col in X.columns:
            if col not in self.covariate_cols:
                reg = LinearRegression().fit(X[self.covariate_cols], X[col])
                self.models_[col] = reg
        return self  # test

    def transform(self, X):
        X_res = X.copy()
        for col, reg in self.models_.items():
            # Subtract the predicted influence of the covariates.
            X_res[col] = X[col] - reg.predict(X[self.covariate_cols])
        return X_res


def _identify_continuous_cols(df, covariate_cols):
    """
    Identify the columns that need to be scaled. We scale continuous or
    quasi-continuous variables that are not covariates. Dichotomous variables
    are not scaled.

    Args:
        df (pd.DataFrame): The cleaned phenotypic dataframe.
        covariate_cols (list): The names of the covariate columns.

    Returns:
        list: The names of the columns that need to be scaled.
    """
    scaling_cols = []
    for col in df.columns:
        if col in covariate_cols:
            continue
        if is_numeric_dtype(df[col]) and df[col].nunique() > 2:
            scaling_cols.append(col)
    return scaling_cols


if __name__ == "__main__":
    # Load the data
    data = pd.read_pickle(BLD_DATA / "encoded_phenotypic_data.pkl")
    targets = pd.read_pickle(BLD_DATA / "clean_cluster_labels.pkl")

    covariates = ["age", "bmi", "sex"]
    mean_score, std_score, result_df = svm_model(data, targets, covariates)
    # result_df.to_csv(BLD_DATA / "cluster5_probabilities.csv")
    # result_df.to_pickle(BLD_DATA / "cluster5_probabilities.pkl")
