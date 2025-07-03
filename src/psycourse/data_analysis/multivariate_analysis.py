import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from psycourse.config import BLD_DATA
from psycourse.ml_pipeline.impute import KNNMedianImputer
from psycourse.ml_pipeline.train_model import DataFrameImputer


def multivariate_analysis(multimodal_data):
    """
    //TODO: update docstring
    Fit a multivariate regression model to the multimodal data using
    regularized regression techniques to predict cluster 5 probability
    Args:
        multimodal_data (pd.DataFrame):
        DataFrame containing multimodal features and target variable (cluster prob).
    Returns:
        model (Pipeline):
    """

    # 1. Data Preparation

    data = multimodal_data
    data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]
    prs_features = [col for col in data.columns if col.endswith("PRS")]
    lipid_features = [col for col in data.columns if col.startswith("gpeak")]  # noqa F841
    lipid_class_features = [col for col in data.columns if col.endswith("mean")]
    # Remove rows where all lipid features are NaN
    data_with_lipids = data[~data[lipid_class_features].isna().all(axis=1)]
    relevant_cols = covariates + lipid_class_features + prs_features + target

    analysis_data = data_with_lipids[relevant_cols]

    # 2. Data Splitting

    X = analysis_data.drop(columns=target).copy()
    y = analysis_data[target].copy()

    y_binary = (y > np.median(y)).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=42
    )

    # 3. Pipeline

    pipeline = Pipeline(
        [
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=7))),
            ("scaler", StandardScaler()),  # TODO: do I scale everything?
            ("variance_threshold", VarianceThreshold(threshold=0.01)),  # Remove low
            ("pca", PCA(n_components=0.95)),
            ("regression", ElasticNetCV(cv=5, random_state=42)),
        ]
    )

    # 4. Nested cross-validation

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)  # noqa F841
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    nested_scores = cross_val_score(
        pipeline, X=X_train, y=y_train, cv=outer_cv, scoring="r2", verbose=True
    )

    print("Mean Nested CV Score: {:.4f}".format(nested_scores.mean()))
    print("Standard Deviation of Nested CV Scores: {:.4f}".format(nested_scores.std()))


if __name__ == "__main__":
    multimodal_df = pd.read_pickle(BLD_DATA / "multimodal_complete_df.pkl")
    multivariate_analysis(multimodal_df)
