from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from psycourse.config import BLD_DATA
from psycourse.ml_pipeline.impute import KNNMedianImputer
from psycourse.ml_pipeline.train_model import DataFrameImputer


def explorative_stage_one_classification(
    multimodal_data, view, cutoff_quantile, n_inner_cv, n_outer_cv, seed, clf_n_jobs
):
    """
    Stage 1: Classify zero vs. non-zero prob_class_5
    Args:
        multimodal_data (pd.DataFrame): DataFrame containing multimodal features and
        target variable (cluster prob).
        cutoff_quantile (float): Quantile to determine the cutoff for classification.
        n_inner_cv (int): Number of inner cross-validation folds
        for hyperparametertuning.
        n_outer_cv (int): Number of outer cross-validation folds for model evaluation.

    Returns:
        model (Pipeline): Trained classification model.
    """

    # 1. Data Preparation

    analysis_data, covariates, target, lipid_features, prs_features = _prepare_data(
        multimodal_data, view
    )

    print(len(analysis_data))
    # 2. Data Splitting
    X = analysis_data.drop(columns=target).copy()
    y = analysis_data[target].copy()

    y_binary = (y > 0).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=seed
    )

    cutoff = y_train["prob_class_5"].quantile(cutoff_quantile)
    y_train_bin = (y_train["prob_class_5"] > cutoff).astype(int).values.ravel()
    y_test_bin = (y_test["prob_class_5"] > cutoff).astype(int).values.ravel()
    print("Using cutoff =", cutoff, cutoff_quantile)  # print-statements for now
    print("Class counts:", np.bincount(y_train_bin))

    # 3. Pre-processing

    # Pre-Processor: only need to standardize lipid features and covariates
    to_scale = []

    if lipid_features:
        to_scale += lipid_features
        to_scale += covariates

    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), to_scale)] if to_scale else [],
        remainder="passthrough",
    )

    # Define classifier
    classifier = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,  # dummy placeholder, will be tuned later
        solver="saga",
        class_weight="balanced",
        max_iter=10_000,
        random_state=seed,
    )

    # Pipeline

    pipeline = Pipeline(
        [
            ("imputer", DataFrameImputer(KNNMedianImputer(n_neighbors=5))),
            ("preprocessing", preprocessor),
            ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ("pca", PCA()),
            ("clf", classifier),
        ]
    )

    # Search space for hyperparameters
    param_distributions = {
        "pca__n_components": [None, 0.99, 0.95],
        "clf__C": loguniform(1e-4, 1e2),
        "clf__l1_ratio": uniform(0, 1),
    }

    # 4. Nested cross-validation

    inner_cv = StratifiedKFold(n_splits=n_inner_cv, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=n_outer_cv, shuffle=True, random_state=seed)

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
    best_params = model.best_params_

    # Model Evaluation on Test Set

    y_pred_bin, y_pred_proba, eval_metrics = _evaluate_classifier(
        best_model, X_test, y_test_bin
    )
    # print("Test Set Evaluation Metrics:", eval_metrics)
    ## Precision: Of all predicted positives, how many were true positives?
    ## Recall = Sensitivity: Of all actual positives,
    # how many were predicted as positive?
    ## F1 Score: Harmonic mean of precision and recall

    # Fit on the entire dataset (X, y):
    final_model = pipeline.set_params(**best_params)
    final_model.fit(X, (y["prob_class_5"] > cutoff).astype(int).values.ravel())

    # Create report
    report = ClassificationReport(
        cutoff_quantile=cutoff_quantile,
        cutoff=cutoff,
        class_counts=np.bincount(y_train_bin),
        classifier="Logistic Regression with ElasticNet",
        inner_cv=n_inner_cv,
        outer_cv=n_outer_cv,
        nested_scores=nested_scores,
        parameters=best_params,
        test_accuracy=eval_metrics["test_accuracy"],
        test_balanced_accuracy=eval_metrics["test_balanced_accuracy"],
        test_avg_precision=eval_metrics["average_precision"],
        test_roc_auc=eval_metrics["test_roc_auc"],
        test_precision=eval_metrics["test_precision"],
        test_recall=eval_metrics["test_recall"],
        confusion_matrix=eval_metrics["confusion_matrix"],
        test_specificity=eval_metrics["specificity"],
        test_mcc=eval_metrics["matthews_corrcoef"],
        test_prevalence=eval_metrics["prevalence"],
    )
    print("Report:", report)
    return final_model, report


def explorative_stage_two_regression(
    multimodal_data, view, cutoff_quantile, n_inner_cv, n_outer_cv, seed, reg_n_jobs
):
    """Stage 2: Regress the magnitude among the non-zeros
    Args:
        multimodal_data (pd.DataFrame): DataFrame containing multimodal features and
        target variable (cluster prob).
        cutoff_quantile (float): Quantile to determine the cutoff for classification.
        n_inner_cv (int): Number of folds for inner cross-validation.
        n_outer_cv (int): Number of folds for outer cross-validation.
    Returns:
        model (Pipeline): Trained regression model.
    """

    # 1. Data Preparation

    analysis_data, covariates, target, lipid_features, prs_features = _prepare_data(
        multimodal_data, view
    )

    # 2. Data Splitting
    X = analysis_data.drop(columns=target).copy()
    y = analysis_data[target].copy()

    y_binary = (y > 0).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=seed
    )

    cutoff = y_train["prob_class_5"].quantile(cutoff_quantile)

    # Filter out non-zero prob_class_5 for regression
    non_zero_mask = y_train["prob_class_5"] > cutoff
    X_train_reg = X_train[non_zero_mask]
    y_train_reg = y_train[non_zero_mask]["prob_class_5"]
    X_test_reg = X_test[y_test["prob_class_5"] > cutoff]
    y_test_reg = y_test[y_test["prob_class_5"] > cutoff]["prob_class_5"]
    print(f"Training set size for regression: {len(X_train_reg)}")
    print(f"Test set size for regression: {len(X_test_reg)}")

    to_scale = []
    if lipid_features:
        to_scale += lipid_features
        to_scale += covariates

    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), to_scale)] if to_scale else [],
        remainder="passthrough",
    )

    # Define regression model

    elastic_net = ElasticNet(max_iter=5000, random_state=42)

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

    inner_cv = KFold(n_splits=n_inner_cv, shuffle=True, random_state=seed)
    outer_cv = KFold(n_splits=n_outer_cv, shuffle=True, random_state=seed)

    # 5. Model Training

    # transform target variable for regression
    ttr = TransformedTargetRegressor(
        regressor=pipeline_regression, func=np.log1p, inverse_func=np.expm1
    )

    # Randomized Search CV for hyperparameter tuningwarp
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
    best_model = model.best_estimator_  # noqa F841
    print("Best inner CV R²: {:.4f}".format(model.best_score_))
    print("Best parameters found: ", model.best_params_)

    # Evaluate on Test Set
    y_pred = model.predict(X_test_reg)
    r2_raw = r2_score(y_test_reg, y_pred)
    mse = mean_squared_error(y_test_reg, y_pred)
    mae = np.mean(np.abs(y_test_reg - y_pred))
    rmse = np.sqrt(mse)

    # Final Model

    # Refit the final model on the entire non-zero training set with best params

    final_model = _get_final_regression_model(
        pipeline_regression, model.best_params_, X_train_reg, y_train_reg
    )

    report = RegressionReport(
        cutoff_quantile=cutoff_quantile,
        cutoff=cutoff,
        train_size=len(X_train_reg),
        test_size=len(X_test_reg),
        inner_cv=n_inner_cv,
        outer_cv=n_outer_cv,
        nested_scores=nested_scores,
        best_parameters=model.best_params_,
        best_inner_cv_r2=model.best_score_,
        test_regression_r2=r2_raw,
        test_regression_mse=mse,
        test_regression_mae=mae,
        test_regression_rmse=rmse,
    )
    print("Regression Report:", report)

    return final_model, report


########################################################################################
######################## HELPER FUNCTIONS ##############################################
########################################################################################


class DataView(str, Enum):
    MULTIMODAL = "multimodal"
    PRS_ONLY = "prs_only"
    LIPIDS_ONLY = "lipids_only"


def _prepare_data(multimodal_data, view):
    view = str(view)

    data = multimodal_data.copy()

    # robust mapping; keeps NA if unexpected code present
    if "sex" in data.columns:
        data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())

    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]

    all_prs = [col for col in data.columns if col.endswith("PRS")]
    all_lipids = [col for col in data.columns if col.startswith("gpeak")]

    # start with no row filtering
    data_sel = data

    if view == "prs_only":
        prs_features = all_prs
        lipid_features = []
        selected_cols = covariates + prs_features + target

    elif view == "lipids_only":
        prs_features = []
        lipid_features = all_lipids
        selected_cols = covariates + lipid_features + target
        if lipid_features:
            # filter ROWS here, keep selected_cols as list of column names
            mask_any_lipid = ~data[lipid_features].isna().all(axis=1)
            data_sel = data.loc[mask_any_lipid]

    elif view == "multimodal":
        prs_features = all_prs
        lipid_features = all_lipids
        selected_cols = covariates + prs_features + lipid_features + target
        if lipid_features:
            mask_any_lipid = ~data[lipid_features].isna().all(axis=1)
            data_sel = data.loc[mask_any_lipid]
    else:
        raise ValueError(f"Unknown view: {view!r}")

    # guard against missing columns after filtering
    selected_cols = [col for col in selected_cols if col in data_sel.columns]

    analysis_data = data_sel[selected_cols].copy()
    return analysis_data, covariates, target, lipid_features, prs_features


def _evaluate_classifier(model, X_test, y_test_bin):
    y_pred_bin = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test_bin, y_pred_bin)
    average_precision = average_precision_score(y_test_bin, y_pred_proba)
    recall = recall_score(y_test_bin, y_pred_bin)
    f1 = f1_score(y_test_bin, y_pred_bin)
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba)
    accuracy = accuracy_score(y_test_bin, y_pred_bin)
    balanced_accuracy = balanced_accuracy_score(y_test_bin, y_pred_bin)
    conf_matrix = confusion_matrix(y_test_bin, y_pred_bin)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    matthews_corr = matthews_corrcoef(y_test_bin, y_pred_bin)
    prevalence = np.mean(y_test_bin)

    eval_metrics = {
        "test_precision": precision,
        "average_precision": average_precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc,
        "test_accuracy": accuracy,
        "test_balanced_accuracy": balanced_accuracy,
        "confusion_matrix": conf_matrix,
        "specificity": specificity,
        "matthews_corrcoef": matthews_corr,
        "prevalence": prevalence,
    }
    return y_pred_bin, y_pred_proba, eval_metrics


def _plot_learning_curve_classifier(best_model, X_train, y_train):
    train_sizes, train_scores, valid_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    # Compute mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores_mean, label="Training Score", marker="o")
    ax.plot(train_sizes, valid_scores_mean, label="Validation Score", marker="o")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("ROC AUC Score")
    ax.legend()
    ax.grid()

    return fig


def _get_feature_importances_classifier(final_model):
    # Get feature importances from the final model
    ## Get best parameters (back-project PCA)

    # Extract components
    logreg = final_model.named_steps["clf"]
    pca = final_model.named_steps["pca"]

    # Coefficients in PCA space
    coef_pca_space = logreg.coef_

    # Back-project to original (pre-PCA) space
    coef_original_space = coef_pca_space @ pca.components_

    # Get absolute values as importance
    importances = np.abs(coef_original_space.ravel())

    # Feature names before PCA (after preprocessing)
    feature_names = final_model.named_steps["preprocessing"].get_feature_names_out()

    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    top20_features = feature_importance_df.head(20)

    return top20_features


def _get_feature_importances_regression(final_model):
    fitted_pipeline = final_model.regressor_
    enet_model = fitted_pipeline.named_steps["enet"]
    pca_model = fitted_pipeline.named_steps["pca"]
    preprocessor = fitted_pipeline.named_steps["preprocessing"]

    # Coefficients in PCA space
    coef_pca_space = enet_model.coef_

    # Back-projection to original feature space
    coef_original_space = coef_pca_space @ pca_model.components_

    # Absolute values as feature importances
    importances = np.abs(coef_original_space)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)
    top20_features = feature_importance_df.head(20)

    return top20_features


def _get_final_regression_model(pipeline, best_params, X_train_reg, y_train_reg):
    final_pipeline = pipeline.set_params(
        **{key.replace("regressor__", ""): val for key, val in best_params.items()}
    )
    final_ttr = TransformedTargetRegressor(
        regressor=final_pipeline, func=np.log1p, inverse_func=np.expm1
    )
    final_ttr.fit(X_train_reg, y_train_reg)
    return final_ttr


def _plot_summary_auc_by_cutoff(df_summary, metric_mean, metric_std):
    grouped_df = (
        df_summary.groupby("cutoff_quantile")[metric_mean, metric_std]
        .mean()
        .reset_index()
    )
    plt.figure()
    plt.errorbar(
        grouped_df["cutoff_quantile"],
        grouped_df[metric_mean],
        yerr=grouped_df[metric_std],
        fmt="o",
        capsize=5,
    )
    plt.title("Summary AUC by Cutoff")
    plt.xlabel("Cutoff Quantile")
    plt.ylabel("AUC")
    plt.grid()
    plt.show()

    return plt


@dataclass
class ClassificationReport:
    cutoff_quantile: float
    cutoff: float
    class_counts: np.ndarray
    classifier: str
    inner_cv: int
    outer_cv: int
    nested_scores: np.ndarray
    parameters: dict
    test_accuracy: float
    test_balanced_accuracy: float
    test_avg_precision: float
    test_mcc: float
    test_specificity: float
    test_prevalence: float
    test_roc_auc: float
    test_precision: float
    test_recall: float
    confusion_matrix: np.ndarray
    # confusion_matrix_figure: plt.Figure
    # precision_recall_figure: plt.Figure
    # roc_figure: plt.Figure
    # learning_curves_figure: plt.Figure
    # top20_features: pd.DataFrame


@dataclass
class RegressionReport:
    cutoff_quantile: float
    cutoff: float
    train_size: int
    test_size: int
    inner_cv: int
    outer_cv: int
    nested_scores: np.ndarray
    best_parameters: dict
    best_inner_cv_r2: float
    test_regression_r2: float
    test_regression_mse: float
    test_regression_mae: float
    test_regression_rmse: float
    # permutation_score: float
    # permutation_pvalue: float
    # learning_curves_figure: plt.Figure
    # top20_features: pd.DataFrame


@dataclass
class TwoStepHurdleReport:
    cutoff_quantile: float
    n_inner_cv: int
    n_outer_cv: int
    n_repeats: int
    test_accs_mean: float
    test_accs_std: float
    test_roc_aucs_mean: float
    test_roc_aucs_std: float
    test_precisions_mean: float
    test_precisions_std: float
    test_recalls_mean: float
    test_recalls_std: float
    test_r2s_mean: float
    test_r2s_std: float
    test_mses_mean: float
    test_mses_std: float
    # permutation_p_vals_mean: float
    # permutation_p_vals_std: float


if __name__ == "__main__":
    multimodal_df = pd.read_pickle(BLD_DATA / "multimodal_complete_df.pkl")
    # explorative_stage_one_classification(
    #    multimodal_df,
    #    view="lipids_only",
    #    cutoff_quantile=0.25,
    #    n_inner_cv=2,
    #    n_outer_cv=2,
    #    seed=42,
    #    clf_n_jobs=-2,
    # )
    explorative_stage_two_regression(
        multimodal_df,
        view="prs_only",
        cutoff_quantile=0.25,
        n_inner_cv=2,
        n_outer_cv=2,
        seed=42,
        reg_n_jobs=-2,
    )
