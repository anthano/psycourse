import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.svm import LinearSVC

from psycourse.ml_pipeline.impute import KNNMedianImputer

###############################################################################
# Main SVM Model Function
###############################################################################


def svm_model(clean_dataset_for_classifier):
    """
    Train an SVM model with nested cross-validation on the given data.

    Args:
        clean_dataset_for_classifier (pd.DataFrame): The dataset to train the model on.
        The dataset should contain features and a target column named "cluster_label".

    Returns:
        model: The trained SVM model after hyperparameter tuning.
        full_df: A DataFrame containing predicted probabilities and labels
        for the entire dataset.
    """
    data = clean_dataset_for_classifier

    X = data.drop("cluster_label", axis=1).copy()
    y = data["cluster_label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Define SVM model:
    base_svm = LinearSVC(
        penalty="l2", loss="hinge", dual=True, class_weight="balanced", max_iter=5000
    )

    # Build the pipeline:
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
    # Define grid search parameters.
    parameters_grid = {
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

    # \\TODO: adjust range from hyperparameter tuning to 2**-6 to 2**6

    # Set up nested cross-validation (10 inner folds and 10 outer folds)
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-2,
        verbose=True,
    )

    # Perform nested cross-validation
    nested_scores = cross_val_score(
        clf,
        X=X_train,
        y=y_train,
        cv=outer_cv,
        scoring="balanced_accuracy",
        n_jobs=-2,
        verbose=True,
    )
    print("Mean Nested CV Score: {:.4f}".format(nested_scores.mean()))

    # Fit the model on training data to get best model
    clf.fit(X_train, y_train)
    print("Best Parameters: ", clf.best_params_)
    print("Best Score: ", clf.best_score_)

    # Retrieve best estimator
    best_model = clf.best_estimator_

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    bac = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy on Test Set: {bac:.4f}")

    # Visualization
    train_sizes, train_scores, valid_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="balanced_accuracy",
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
    plt.ylabel("Balanced Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    # plt.show()

    # Obtain predicted probabilities for each class
    y_score = best_model.predict_proba(X_test)

    # Get the unique classes from y_test
    classes = np.unique(y_test)
    n_classes = len(classes)

    # Binarize the output for multiclass ROC analysis
    y_test_bin = label_binarize(y_test, classes=classes)

    # Dictionaries to store FPR, TPR and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure(figsize=(8, 6))
    colors = ["blue", "red", "green", "orange", "purple"]
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=2,
            label="Class {0} (AUC = {1:0.2f})".format(classes[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Each Class")
    plt.legend(loc="lower right")
    plt.show()

    # Compute and display the confusion matrix
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Get predicted probabilities
    probabilities = best_model.predict_proba(X_test)

    # Class labels from the model
    class_labels = best_model.classes_

    # Create DataFrame explicitly preserving the original index
    prob_df = pd.DataFrame(
        probabilities,
        columns=[f"prob_class_{cls}" for cls in class_labels],
        index=X_test.index,
    )

    prob_df["true_label"] = y_test
    prob_df["predicted_label"] = best_model.predict(X_test)

    # Refit the model on the entire dataset (X, y):

    # Unpack best hyperparameters:
    best_params = clf.best_params_

    # Reconfigure the pipeline with those params:
    final_model = pipeline.set_params(**best_params)

    # Fit on the entire dataset (X, y):
    final_model.fit(X, y)

    # Predict probabilities and labels on the full X:
    probs_full = final_model.predict_proba(X)
    preds_full = final_model.predict(X)

    # Build a DataFrame with the original index:
    full_df = pd.DataFrame(
        probs_full,
        index=X.index,
        columns=[f"prob_class_{cls}" for cls in final_model.classes_],
    )
    full_df["true_label"] = y
    full_df["predicted_label"] = preds_full
    full_df["predicted_label"] = full_df["predicted_label"].astype(
        pd.CategoricalDtype()
    )
    print(full_df.head())

    return final_model, full_df


###############################################################################
# Helper Transformers & Functions
###############################################################################


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    A wrapper for an imputer (e.g. KNNImputer) that ensures the output remains
    a DataFrame with the same columns and index as the input.
    """

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None):
        # Store column names and index if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            self.index_ = X.index
            self.dtypes_ = X.dtypes
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_t = self.imputer.transform(X)
        # If the input was originally a DataFrame, convert back to DataFrame
        if hasattr(self, "columns_"):
            imputed = pd.DataFrame(
                X_t,
                columns=self.columns_,
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )
            # Handle integer dtypes
            int_dtypes = [
                pd.Int8Dtype(),
                pd.Int16Dtype(),
                pd.Int32Dtype(),
                pd.Int64Dtype(),
            ]
            for col in self.columns_[self.dtypes_.isin(int_dtypes)]:
                imputed[col] = imputed[col].round()
            return imputed.astype(self.dtypes_)
        return X_t
