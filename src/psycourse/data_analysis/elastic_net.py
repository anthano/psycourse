import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    learning_curve,
    permutation_test_score,
    train_test_split,
)
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

    # TODO: move this to a separate function?

    data = multimodal_data
    data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]
    prs_features = [col for col in data.columns if col.endswith("PRS")]
    lipid_features = [col for col in data.columns if col.startswith("gpeak")]
    # Remove rows where all lipid features are NaN
    data_with_lipids = data[~data[lipid_features].isna().all(axis=1)]
    relevant_cols = covariates + lipid_features + prs_features + target

    analysis_data = data_with_lipids[relevant_cols].copy()

    # 2. Data Splitting

    X = analysis_data.drop(columns=target).copy()
    # y = np.log(analysis_data[target]).copy()
    y = analysis_data[target].copy()  # use raw probability for regression

    y_binary = (y > np.median(y)).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=42
    )

    # 3. Model Definition

    elastic_net = ElasticNet(max_iter=1000, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), lipid_features + covariates)],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=7))),
            ("preprocessing", preprocessor),
            ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ("pca", PCA()),
            ("regression", elastic_net),
        ]
    )

    param_distributions = {
        "regressor__pca__n_components": [None, 0.99, 0.95],
        "regressor__regression__alpha": loguniform(1e-4, 1e2),
        "regressor__regression__l1_ratio": uniform(0, 1),
    }

    # 4. Nested cross-validation

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    ttr = TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1
    )

    model = RandomizedSearchCV(
        ttr,
        param_distributions=param_distributions,
        n_iter=50,
        cv=inner_cv,
        scoring="r2",
        verbose=1,
        n_jobs=-2,
        error_score="raise",
    )

    nested_scores = cross_val_score(
        model, X=X_train, y=y_train, cv=outer_cv, scoring="r2", verbose=True
    )

    print("Mean Nested CV R2: {:.4f}".format(nested_scores.mean()))
    print("Standard Deviation of Nested CV Scores: {:.4f}".format(nested_scores.std()))

    model.fit(X_train, y_train)
    best_model = model.best_estimator_
    print("Best inner CV R²: {:.4f}".format(model.best_score_))
    print("Best parameters found: ", model.best_params_)

    # Retrieve the best model
    # best_model = model.best_estimator_

    # Evaluate on Test Set
    y_pred = model.predict(X_test)
    r2_raw = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test R² (raw): {r2_raw:.4f} Test MSE: {mse:.4f}")

    # Null R2
    y0 = y_train[target[0]].mean()
    null_pred = np.full(len(y_test), y0)
    print("Null R²:", r2_score(y_test, null_pred))

    # Visualization
    plt = _plot_learning_curve(best_model, X_train, y_train)
    plt.show()

    # Permutation Test to test significance of the model
    y_train_array = y_train.values.ravel()  # Ensure y_train is a 1D array
    score, permutation_scores, pvalue = permutation_test_score(
        estimator=best_model,
        X=X_train,
        y=y_train_array,
        cv=outer_cv,
        n_permutations=1000,
        scoring="r2",
        n_jobs=-2,
        random_state=42,
    )

    print(f"Observed R²    : {score:.4f}")
    print(
        f"Null R² 5th–95th : "
        f"{np.percentile(permutation_scores, 5):.3f} – "
        f"{np.percentile(permutation_scores, 95):.3f}"
    )
    print(f"P-value (one-sided): {pvalue:.3f}")


############################### Helper Functions ####################################


def _plot_learning_curve(best_model, X_train, y_train):
    train_sizes, train_scores, valid_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-2,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    # Compute mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(8, 6))
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, valid_scores_mean, "o-", color="g", label="Validation score")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("R2")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.show()

    return plt


if __name__ == "__main__":
    multimodal_df = pd.read_pickle(BLD_DATA / "multimodal_complete_df.pkl")
    multivariate_analysis(multimodal_df)
