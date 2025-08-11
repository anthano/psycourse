import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
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


def two_step_hurdle(multimodal_data, n_repeats=2, base_seed=42):
    """
    Perform two-stage hurdle model on multimodal data to predict cluster 5 probability.
    Args:
    Returns:
    """

    # cutoff_quantile = [0.05, 0.10, 0.25]
    # n_inner_cv = [2, 5, 10]
    # n_outer_cv = [2, 5, 10]

    cutoff_quantile = [0.05]
    n_inner_cv = [2]
    n_outer_cv = [2]

    results = []

    for cutoff in cutoff_quantile:
        for inner_cv in n_inner_cv:
            for outer_cv in n_outer_cv:
                (
                    test_accs,
                    test_roc_aucs,
                    test_precisions,
                    test_recalls,
                    top20_features_clf,
                ) = [], [], [], [], []
                test_r2s, test_mses, permutation_p_vals, top20_features_reg = (
                    [],
                    [],
                    [],
                    [],
                )
                combo_id = f"{cutoff}_{inner_cv}_{outer_cv}"
                combo_hash = (
                    int(hashlib.sha256(combo_id.encode()).hexdigest(), 16) % 1_000_000
                )
                print(combo_id)
                for repeat in range(n_repeats):
                    seed = base_seed + combo_hash + repeat
                    clf_model, clf_report = stage_one_classification(
                        multimodal_data, cutoff, inner_cv, outer_cv, seed
                    )
                    print("Stage 1 Classification Report:", clf_report)

                    reg_model, reg_report = stage_two_regression(
                        multimodal_data, cutoff, inner_cv, outer_cv, seed
                    )
                    print("Stage 2 Regression Report:", reg_report)

                    test_accs.append(clf_report.test_accuracy)
                    test_roc_aucs.append(clf_report.test_roc_auc)
                    test_precisions.append(clf_report.test_precision)
                    test_recalls.append(clf_report.test_recall)
                    top20_features_clf.append(clf_report.top20_features)

                    test_r2s.append(reg_report.test_regression_r2)
                    test_mses.append(reg_report.test_regression_mse)
                    permutation_p_vals.append(reg_report.permutation_pvalue)
                    top20_features_reg.append(reg_report.top20_features)

                    results.append(
                        TwoStepHurdleReport(
                            cutoff_quantile=cutoff,
                            n_inner_cv=inner_cv,
                            n_outer_cv=outer_cv,
                            n_repeats=n_repeats,
                            test_accs_mean=np.mean(test_accs),
                            test_accs_std=np.std(test_accs),
                            test_roc_aucs_mean=np.mean(test_roc_aucs),
                            test_roc_aucs_std=np.std(test_roc_aucs),
                            test_precisions_mean=np.mean(test_precisions),
                            test_precisions_std=np.std(test_precisions),
                            test_recalls_mean=np.mean(test_recalls),
                            test_recalls_std=np.std(test_recalls),
                            test_r2s_mean=np.mean(test_r2s),
                            test_r2s_std=np.std(test_r2s),
                            test_mses_mean=np.mean(test_mses),
                            test_mses_std=np.std(test_mses),
                            permutation_p_vals_mean=np.mean(permutation_p_vals),
                            permutation_p_vals_std=np.std(permutation_p_vals),
                        )
                    )

                    print(results)


def stage_one_classification(
    multimodal_data, cutoff_quantile, n_inner_cv, n_outer_cv, seed
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
        multimodal_data
    )

    # 2. Data Splitting
    X = analysis_data.drop(columns=target).copy()
    y = analysis_data[target].copy()

    y_binary = (y > 0).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=42
    )

    cutoff = y_train["prob_class_5"].quantile(cutoff_quantile)  # TODO: very arbitrary
    y_train_bin = (y_train["prob_class_5"] > cutoff).astype(int).values.ravel()
    y_test_bin = (y_test["prob_class_5"] > cutoff).astype(int).values.ravel()
    print("Using cutoff =", cutoff, cutoff_quantile)  # print-statements for now
    print("Class counts:", np.bincount(y_train_bin))

    # 3. Pre-processing

    # Pre-Processor: only need to standardize lipid features and covariates
    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), lipid_features + covariates)],
        remainder="passthrough",
    )

    # Define classifier
    classifier = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,  # dummy placeholder, will be tuned later
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
        n_jobs=-3,
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

    # Plotting

    conf_mat_figure = ConfusionMatrixDisplay.from_predictions(y_test_bin, y_pred_bin)
    precision_recall_figure = PrecisionRecallDisplay.from_predictions(
        y_test_bin, y_pred_proba
    )
    roc_figure = RocCurveDisplay.from_predictions(y_test_bin, y_pred_proba)
    # ROC AUC: Area under the ROC curve, a measure of model's ability to
    # distinguish between classes
    learning_curves_figure = _plot_learning_curve_classifier(
        best_model, X_train, y_train_bin
    )
    plt.close(conf_mat_figure.figure_)
    plt.close(precision_recall_figure.figure_)
    plt.close(roc_figure.figure_)
    plt.close(learning_curves_figure)

    # Fit on the entire dataset (X, y):
    final_model = pipeline.set_params(**best_params)
    final_model.fit(X, (y["prob_class_5"] > cutoff).astype(int).values.ravel())

    # Get feature importances from the final model
    top20_features = _get_feature_importances_classifier(final_model)

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
        test_roc_auc=eval_metrics["test_roc_auc"],
        test_precision=eval_metrics["test_precision"],
        test_recall=eval_metrics["test_recall"],
        confusion_matrix=eval_metrics["confusion_matrix"],
        confusion_matrix_figure=conf_mat_figure.figure_,
        precision_recall_figure=precision_recall_figure.figure_,
        roc_figure=roc_figure.figure_,
        learning_curves_figure=learning_curves_figure,
        top20_features=top20_features,
    )
    print("Report:", report)
    return final_model, report


def stage_two_regression(
    multimodal_data, cutoff_quantile, n_inner_cv, n_outer_cv, seed
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
        multimodal_data
    )

    # 2. Data Splitting
    X = analysis_data.drop(columns=target).copy()
    y = analysis_data[target].copy()

    y_binary = (y > 0).astype(int)  # binary as helper for stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binary, random_state=42
    )

    cutoff = y_train["prob_class_5"].quantile(cutoff_quantile)  # TODO: very arbitrary

    # Filter out non-zero prob_class_5 for regression
    non_zero_mask = y_train["prob_class_5"] > cutoff
    X_train_reg = X_train[non_zero_mask]
    y_train_reg = y_train[non_zero_mask]["prob_class_5"]
    X_test_reg = X_test[y_test["prob_class_5"] > cutoff]
    y_test_reg = y_test[y_test["prob_class_5"] > cutoff]["prob_class_5"]
    print(f"Training set size for regression: {len(X_train_reg)}")
    print(f"Test set size for regression: {len(X_test_reg)}")

    # Pre-Processor #TODO: can I get this more elegantly from previous step?
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
        n_jobs=-3,
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

    # Visualization
    learning_curve_fig = _plot_learning_curve(best_model, X_train_reg, y_train_reg)
    # plt.close(learning_curve_fig)

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

    # Final Model

    # Refit the final model on the entire non-zero training set with best params

    final_model = _get_final_regression_model(
        pipeline_regression, model.best_params_, X_train_reg, y_train_reg
    )

    # Get feature importances from the final model

    # Back-project coefficients from PCA space
    top20_features = _get_feature_importances_regression(final_model)
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
        permutation_score=score,
        permutation_pvalue=pvalue,
        learning_curves_figure=learning_curve_fig,
        top20_features=top20_features,
    )
    print("Regression Report:", report)

    return final_model, report


########################################################################################
######################## HELPER FUNCTIONS ##############################################
########################################################################################


def _prepare_data(multimodal_data):
    data = multimodal_data.copy()

    data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]
    prs_features = [col for col in data.columns if col.endswith("PRS")]
    lipid_features = [col for col in data.columns if col.startswith("gpeak")]
    data_with_lipids = data[~data[lipid_features].isna().all(axis=1)]
    relevant_cols = covariates + lipid_features + prs_features + target

    analysis_data = data_with_lipids[relevant_cols].copy()

    return analysis_data, covariates, target, lipid_features, prs_features


def _evaluate_classifier(model, X_test, y_test_bin):
    y_pred_bin = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test_bin, y_pred_bin)
    recall = recall_score(y_test_bin, y_pred_bin)
    f1 = f1_score(y_test_bin, y_pred_bin)
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba)
    accuracy = accuracy_score(y_test_bin, y_pred_bin)
    conf_matrix = confusion_matrix(y_test_bin, y_pred_bin)

    eval_metrics = {
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc,
        "test_accuracy": accuracy,
        "confusion_matrix": conf_matrix,
    }
    return y_pred_bin, y_pred_proba, eval_metrics


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
    test_roc_auc: float
    test_precision: float
    test_recall: float
    confusion_matrix: np.ndarray
    confusion_matrix_figure: plt.Figure
    precision_recall_figure: plt.Figure
    roc_figure: plt.Figure
    learning_curves_figure: plt.Figure
    top20_features: pd.DataFrame


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
    permutation_score: float
    permutation_pvalue: float
    learning_curves_figure: plt.Figure
    top20_features: pd.DataFrame


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
    permutation_p_vals_mean: float
    permutation_p_vals_std: float


if __name__ == "__main__":
    multimodal_df = pd.read_pickle(BLD_DATA / "multimodal_complete_df.pkl")
    # stage_one_classification(
    #    multimodal_df, cutoff_quantile=0.05, n_inner_cv=2, n_outer_cv=2
    # )
    # stage_two_regression(
    #    multimodal_df, cutoff_quantile=0.05, n_inner_cv=2, n_outer_cv=2
    # )

    two_step_hurdle(multimodal_df)
