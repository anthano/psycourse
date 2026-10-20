import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from psycourse.ml_pipeline.impute import KNNMedianImputer
from psycourse.ml_pipeline.train_model import DataFrameImputer

###############################################################################
# Out-of-fold SVM probabilities
###############################################################################

# Same pipeline and hyperparameter grid as svm_model in train_model.py. Built here
# rather than imported so that computing out-of-fold predictions does not retrain
# the primary model. LinearSVC is seeded for reproducibility.
PARAMETERS_GRID = {
    "pca__n_components": [0.25, 0.5, 0.75, 1],
    "svm__estimator__estimator__C": [
        0.015625,
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        1,
        2,
        4,
        8,
        16,
    ],
}


def build_svm_search(random_state=42, n_jobs=-2):
    """
    Grid search over the SVM pipeline used for the severe psychosis subtype.

    Args:
        random_state (int): Seed for LinearSVC and the inner cross-validation.
        n_jobs (int): Parallel jobs for the grid search.
    Returns:
        GridSearchCV: Unfitted grid search with 10-fold inner cross-validation.
    """
    base_svm = LinearSVC(
        penalty="l2",
        loss="hinge",
        dual=True,
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    pipeline = Pipeline(
        [
            ("pruning", VarianceThreshold(threshold=0.0)),
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=7))),
            (
                "first_scaling",
                Pipeline(
                    [
                        ("scaler", MinMaxScaler(feature_range=(-1, 1))),
                    ]
                ),
            ),
            ("pca", PCA()),
            ("second_scaling", MinMaxScaler(feature_range=(-1, 1))),
            (
                "svm",
                CalibratedClassifierCV(estimator=OneVsOneClassifier(base_svm), cv=5),
            ),
        ]
    )
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=PARAMETERS_GRID,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=n_jobs,
    )


def svm_out_of_fold_probabilities(
    clean_dataset_for_classifier, n_splits=10, random_state=42, n_jobs=-2
):
    """
    Out-of-fold class probabilities from nested cross-validation.

    Each participant's probabilities come from a model that was tuned (inner 10-fold
    grid search) and fitted without that participant (outer folds), so the scores
    carry no in-sample optimism.

    Args:
        clean_dataset_for_classifier (pd.DataFrame): Features and "cluster_label".
        n_splits (int): Number of outer folds.
        random_state (int): Seed for the outer folds and the grid search.
        n_jobs (int): Parallel jobs for the grid search.
    Returns:
        oof_df (pd.DataFrame): prob_class_* columns, true_label, predicted_label,
        and outer_fold, indexed like the input.
        fold_summary (pd.DataFrame): Per outer fold: n_test, best hyperparameters,
        and balanced accuracy on the held-out fold.
    """
    X = clean_dataset_for_classifier.drop("cluster_label", axis=1).copy()
    y = clean_dataset_for_classifier["cluster_label"].copy()
    classes = np.sort(y.unique())

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probabilities = np.full((len(X), len(classes)), np.nan)
    outer_fold = np.zeros(len(X), dtype=int)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        search = build_svm_search(random_state=random_state, n_jobs=n_jobs)
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        assert list(search.classes_) == list(classes)

        probabilities[test_idx] = search.predict_proba(X.iloc[test_idx])
        outer_fold[test_idx] = fold
        rows.append(
            {
                "fold": fold,
                "n_test": len(test_idx),
                **search.best_params_,
                "balanced_accuracy": balanced_accuracy_score(
                    y.iloc[test_idx], search.predict(X.iloc[test_idx])
                ),
            }
        )

    oof_df = pd.DataFrame(
        probabilities,
        index=X.index,
        columns=[f"prob_class_{cls}" for cls in classes],
    )
    oof_df["true_label"] = y
    oof_df["predicted_label"] = pd.Categorical(classes[probabilities.argmax(axis=1)])
    oof_df["outer_fold"] = outer_fold

    fold_summary = pd.DataFrame(rows).set_index("fold")
    return oof_df, fold_summary
