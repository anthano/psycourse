import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    permutation_test_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from psycourse.config import BLD_DATA
from psycourse.data_analysis.elastic_net import _plot_learning_curve
from psycourse.ml_pipeline.impute import KNNMedianImputer
from psycourse.ml_pipeline.train_model import DataFrameImputer


def stage_one_classification_rf(multimodal_data):
    """
    Stage 1: Classify zero vs. non-zero prob_class_5
    Args:
        multimodal_data (pd.DataFrame): DataFrame containing multimodal features and
        target variable (cluster prob).
    Returns:
        model (Pipeline): Trained classification model.
    """

    # 1. Data Preparation
    data = multimodal_data.copy()
    data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]
    prs_features = [col for col in data.columns if col.endswith("PRS")]
    lipid_features = [col for col in data.columns if col.startswith("gpeak")]
    analysis_data = _prepare_data(
        data, target, covariates, lipid_features, prs_features
    )

    # 2. Data Splitting
    X = analysis_data.drop(columns=target).copy()
    y = analysis_data[target].copy()

    y_binary = (y > 0).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=42
    )

    cutoff = y_train["prob_class_5"].quantile(0.25)  # TODO: very arbitrary
    y_train_bin = (y_train["prob_class_5"] > cutoff).astype(int).values.ravel()
    print("Using cutoff =", cutoff)
    print("Class counts:", np.bincount(y_train_bin))

    # 3. Pre-processing and Model Definition

    # Pre-Processer
    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), lipid_features + covariates)],
        remainder="passthrough",
    )

    classifier = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        class_weight="balanced",
        max_iter=10000,
        random_state=42,
    )

    # Pipeline

    pipeline = Pipeline(
        [
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=7))),
            ("preprocessing", preprocessor),
            ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ("pca", PCA()),
            ("clf", classifier),
        ]
    )

    # Parameters for Search
    param_distributions = {
        "pca__n_components": [None, 0.99, 0.95],
        "clf__C": loguniform(1e-4, 1e2),
        "clf__l1_ratio": uniform(0, 1),
    }

    # 4. Nested cross-validation

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5. Model Training

    model = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        cv=inner_cv,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-2,
        error_score="raise",
    )

    # Hyperparameter Tuning in Nested CV
    nested_scores = cross_val_score(
        model,
        X_train,
        y_train_bin,
        cv=outer_cv,
        scoring="roc_auc",
        verbose=True,
        error_score="raise",
    )

    # Model Fitting

    model.fit(X_train, y_train_bin)
    best_model = model.best_estimator_

    # Model Evaluation on Test Set

    y_test_bin = (y_test["prob_class_5"] > cutoff).astype(int).values
    y_pred_bin = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test_bin, y_pred_bin)
    test_auc = roc_auc_score(y_test_bin, y_pred_proba)
    test_precision = precision_score(y_test_bin, y_pred_bin)
    test_recall = recall_score(y_test_bin, y_pred_bin)

    conf_mat = confusion_matrix(y_test_bin, y_pred_bin)

    PrecisionRecallDisplay.from_predictions(y_test_bin, y_pred_proba)
    plt.show()

    ## Plot
    base_pipeline = model.estimator  # unfitted pipeline
    best_params = model.best_params_

    pipeline_with_best_params = base_pipeline.set_params(**best_params)

    learning_curves_plot = _plot_learning_curve_classifier(
        pipeline_with_best_params, X_train, y_train_bin
    )
    learning_curves_plot.show()

    report = {
        "cutoff": cutoff,
        "class_counts": np.bincount(y_train_bin),
        "classifier": "RandomForestClassifier",
        "Inner CV": inner_cv,
        "Outer CV": outer_cv,
        "nested_scores": nested_scores,
        "parameters": model.best_params_,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "confusion_matrix": conf_mat,
    }

    print("Report:", report)

    return report


def stage_two_regression(multimodal_data):
    """
    Stage 2: Regress the magnitude among the non-zeros
    Args:
        multimodal_data (pd.DataFrame): DataFrame containing multimodal features and
        target variable (cluster prob).
    Returns:
        model (Pipeline): Trained regression model.
    """

    # 1. Data Preparation
    data = multimodal_data.copy()
    data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]
    prs_features = [col for col in data.columns if col.endswith("PRS")]
    lipid_features = [col for col in data.columns if col.startswith("gpeak")]
    analysis_data = _prepare_data(
        data, target, covariates, lipid_features, prs_features
    )

    # 2. Data Splitting
    X = analysis_data.drop(columns=target).copy()
    y = analysis_data[target].copy()

    y_binary = (y > 0).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=42
    )

    cutoff = y_train["prob_class_5"].quantile(0.25)  # TODO: very arbitrary
    y_train_bin = (y_train["prob_class_5"] > cutoff).astype(int).values.ravel()
    print("Using cutoff =", cutoff)
    print("Class counts:", np.bincount(y_train_bin))

    # Filter out non-zero prob_class_5 for regression
    non_zero_mask = y_train["prob_class_5"] > cutoff
    X_train_reg = X_train[non_zero_mask]
    y_train_reg = y_train[non_zero_mask]["prob_class_5"]
    X_test_reg = X_test[y_test["prob_class_5"] > cutoff]
    y_test_reg = y_test[y_test["prob_class_5"] > cutoff]["prob_class_5"]
    print(f"Training set size for regression: {len(X_train_reg)}")
    print(f"Test set size for regression: {len(X_test_reg)}")

    # Pre-Processer #TODO: can I get this more elegantly from previous step?
    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), lipid_features + covariates)],
        remainder="passthrough",
    )

    # Define regression model

    elastic_net = ElasticNet(max_iter=10000, random_state=42)

    pipeline_regression = Pipeline(
        [
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=7))),
            ("preprocessing", preprocessor),
            ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ("pca", PCA()),
            ("enet", elastic_net),
        ]
    )

    param_distributions_regression = {
        "regressor__pca__n_components": [None, 0.99, 0.95],
        "regressor__enet__alpha": loguniform(1e-4, 1e2),
        "regressor__enet__l1_ratio": uniform(0, 1),
    }

    # 4. Nested cross-validation

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    ttr = TransformedTargetRegressor(
        regressor=pipeline_regression, func=np.log1p, inverse_func=np.expm1
    )

    model = RandomizedSearchCV(
        ttr,
        param_distributions=param_distributions_regression,
        n_iter=50,
        cv=inner_cv,
        scoring="r2",
        verbose=1,
        n_jobs=-2,
        error_score="raise",
    )

    nested_scores = cross_val_score(
        model, X=X_train_reg, y=y_train_reg, cv=outer_cv, scoring="r2", verbose=True
    )

    print("Mean Nested CV R2: {:.4f}".format(nested_scores.mean()))
    print("Standard Deviation of Nested CV Scores: {:.4f}".format(nested_scores.std()))

    model.fit(X_train_reg, y_train_reg)
    best_model = model.best_estimator_
    print("Best inner CV R²: {:.4f}".format(model.best_score_))
    print("Best parameters found: ", model.best_params_)

    # Evaluate on Test Set
    y_pred = model.predict(X_test_reg)
    r2_raw = r2_score(y_test_reg, y_pred)
    mse = mean_squared_error(y_test_reg, y_pred)
    print(f"Test R² (raw): {r2_raw:.4f} Test MSE: {mse:.4f}")

    # Visualization
    plt = _plot_learning_curve(best_model, X_train_reg, y_train_reg)
    plt.show()

    ## Permutation Test to test significance of the model
    y_train_array = y_train_reg.values.ravel()  # Ensure y_train is a 1D array
    score, permutation_scores, pvalue = permutation_test_score(
        estimator=best_model,
        X=X_train_reg,
        y=y_train_array,
        cv=outer_cv,
        n_permutations=1000,
        scoring="r2",
        n_jobs=-2,
        random_state=42,
    )

    metrics = {
        "mean_nested_cv_r2": nested_scores.mean(),
        "std_nested_cv_r2": nested_scores.std(),
        "best_inner_cv_r2": model.best_score_,
        "best_parameters": model.best_params_,
        "test_regression_r2": r2_raw,
        "test_regression_mse": mse,
        "permutation_score": score,
        "permutation_pvalue": pvalue,
    }

    print("Metrics:", metrics)


########################################################################################
######################## HELPER FUNCTIONS ##############################################
########################################################################################


def _prepare_data(multimodal_data, target, covariates, lipid_features, prs_features):
    data = multimodal_data.copy()
    data_with_lipids = data[~data[lipid_features].isna().all(axis=1)]
    relevant_cols = covariates + lipid_features + prs_features + target

    analysis_data = data_with_lipids[relevant_cols].copy()

    return analysis_data


def _plot_learning_curve_classifier(best_model, X_train, y_train):
    train_sizes, train_scores, valid_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-2,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    # Compute mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training Score", marker="o")
    plt.plot(train_sizes, valid_scores_mean, label="Validation Score", marker="o")
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("ROC AUC Score")
    plt.legend()
    plt.grid()

    return plt


if __name__ == "__main__":
    multimodal_df = pd.read_pickle(BLD_DATA / "multimodal_complete_df.pkl")
    stage_two_regression(multimodal_df)
