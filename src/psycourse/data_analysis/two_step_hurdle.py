import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from psycourse.config import BLD_DATA
from psycourse.ml_pipeline.impute import KNNMedianImputer
from psycourse.ml_pipeline.train_model import DataFrameImputer


def hurdle_multivariate_analysis(multimodal_data):
    """
    #TODO: write proper docstring
    Two‐stage (“hurdle”) model:
      1) Classify zero vs. non‐zero prob_class_5
      2) Regress the magnitude among the non‐zeros
    Returns:
      clf, reg, X_test, y_test, y_pred_hurdle
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

    y_binary = (y > 0).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=42
    )

    cutoff = y_train["prob_class_5"].quantile(0.5)

    # Option B: first quartile
    # cutoff = y_train["prob_class_5"].quantile(0.25)

    # Option C: fixed epsilon
    # cutoff = 1e-3

    y_train_bin = (y_train["prob_class_5"] > cutoff).astype(int).values.ravel()
    print("Using cutoff =", cutoff)
    print("Class counts:", np.bincount(y_train_bin))

    # 3. Model Definition

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

    pipeline = Pipeline(
        [
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=7))),
            ("preprocessing", preprocessor),
            ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ("pca", PCA()),
            ("clf", classifier),
        ]
    )

    param_distributions = {
        "pca__n_components": [None, 0.99, 0.95],
        "clf__C": loguniform(1e-4, 1e2),
        "clf__l1_ratio": uniform(0, 1),
    }

    # 4. Nested cross-validation
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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

    # fit on binary target
    # y_train_bin = (y_train[target[0]] > 0).astype(int).values.ravel()
    nested_scores = cross_val_score(
        model,
        X_train,
        y_train_bin,
        cv=outer_cv,
        scoring="roc_auc",
        verbose=True,
        error_score="raise",
    )
    print(
        f"Nested CV accuracy for binary: {nested_scores.mean():.3f}"
        f" ± {nested_scores.std():.3f}"
    )

    model.fit(X_train, y_train_bin)
    best_model = model.best_estimator_
    print("Best model parameters:", model.best_params_)
    print("Best inner CV ROC AUC: {:.4f}".format(model.best_score_))

    # Evaluate on Test Set
    y_test_bin = (y_test["prob_class_5"] > cutoff).astype(int).values
    y_pred_bin = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = model.score(X_test, y_pred_bin)
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # 2) predictions & probabilities
    y_pred_bin = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 3) correct metrics
    test_auc = roc_auc_score(y_test_bin, y_pred_proba)
    test_acc = accuracy_score(y_test_bin, y_pred_bin)
    conf_mat = confusion_matrix(y_test_bin, y_pred_bin)

    print(f"Test ROC AUC : {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Confusion matrix (at 0.5):\n", conf_mat)


if __name__ == "__main__":
    multimodal_df = pd.read_pickle(BLD_DATA / "multimodal_complete_df.pkl")
    hurdle_multivariate_analysis(multimodal_df)
