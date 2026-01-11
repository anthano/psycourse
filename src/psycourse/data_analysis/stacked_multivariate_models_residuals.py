import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def additive_logit_lipid_correction(
    df: pd.DataFrame,
    random_state: int = 42,
    outer_splits: int = 5,
    inner_splits: int = 5,
):
    df = df.copy()

    covariates = [
        "age",
        "sex",
        "smoker",
        "duration_illness",
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pc5",
    ]
    prs_features = ["SCZ_PRS", "MDD_PRS", "BD_PRS"]
    lipid_class_features = [
        "class_LPE",
        "class_PC",
        "class_PC_O",
        "class_PC_P",
        "class_PE",
        "class_PE_P",
        "class_TAG",
        "class_dCer",
        "class_dSM",
        "class_CAR",
        "class_CE",
        "class_DAG",
        "class_FA",
        "class_LPC",
        "class_LPC_O",
        "class_LPC_P",
    ]

    df["sex"] = df["sex"].map({"F": 0, "M": 1})
    df["smoker"] = df["smoker"].map({"former": 0, "never": 1, "yes": 2})

    base_required = ["true_label"] + covariates + prs_features
    keep_mask = df[base_required].notna().all(axis=1)
    keep_mask &= ~df[lipid_class_features].isna().all(
        axis=1
    )  # require at least some lipid class info
    df = df.loc[keep_mask].reset_index(drop=True)

    y_binary = (df["true_label"] == 5).astype(int).to_numpy()

    base_feature_columns = covariates + prs_features

    base_classifier = LogisticRegressionCV(
        Cs=np.logspace(-4, 3, 15),
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
        class_weight="balanced",
        scoring="average_precision",
        cv=inner_splits,
        max_iter=5000,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    base_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", base_classifier),
        ]
    )

    lipid_classifier = LogisticRegressionCV(
        Cs=np.logspace(-4, 3, 15),
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
        class_weight="balanced",
        scoring="average_precision",
        cv=inner_splits,
        max_iter=5000,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    lipid_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", lipid_classifier),
        ]
    )

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv_for_oof = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )

    oof_probability_base = np.zeros(len(df), dtype=float)
    oof_probability_additive = np.zeros(len(df), dtype=float)

    for train_indices, test_indices in outer_cv.split(df, y_binary):
        train_data_outer = df.iloc[train_indices]
        test_data_outer = df.iloc[test_indices]

        y_train_outer = y_binary[train_indices]

        X_base_train_outer = train_data_outer[base_feature_columns]
        X_base_test_outer = test_data_outer[base_feature_columns]

        X_lipid_train_outer = train_data_outer[lipid_class_features]
        X_lipid_test_outer = test_data_outer[lipid_class_features]

        # (1) base: OOF preds on outer-train -> logits
        base_oof_prob_outer_train = cross_val_predict(
            clone(base_pipeline),
            X_base_train_outer,
            y_train_outer,
            cv=inner_cv_for_oof,
            method="predict_proba",
        )[:, 1]
        base_oof_logit_outer_train = _safe_logit(base_oof_prob_outer_train)  # noqa: F841

        # (2) base: fit on outer-train -> preds on outer-test -> logits
        fitted_base_outer_train = clone(base_pipeline).fit(
            X_base_train_outer, y_train_outer
        )
        base_prob_outer_test = fitted_base_outer_train.predict_proba(X_base_test_outer)[
            :, 1
        ]
        base_logit_outer_test = _safe_logit(base_prob_outer_test)
        oof_probability_base[test_indices] = base_prob_outer_test

        # (3) lipid correction:
        # train on (y vs lipids) within outer-train using inner CV OOF logits
        lipid_oof_prob_outer_train = cross_val_predict(
            clone(lipid_pipeline),
            X_lipid_train_outer,
            y_train_outer,
            cv=inner_cv_for_oof,
            method="predict_proba",
        )[:, 1]
        lipid_oof_logit_outer_train = _safe_logit(lipid_oof_prob_outer_train)

        # fit lipid model on full outer-train and predict outer-test -> lipid logit
        fitted_lipid_outer_train = clone(lipid_pipeline).fit(
            X_lipid_train_outer, y_train_outer
        )
        lipid_prob_outer_test = fitted_lipid_outer_train.predict_proba(
            X_lipid_test_outer
        )[:, 1]
        lipid_logit_outer_test = _safe_logit(lipid_prob_outer_test)

        # (4) combine additively in logit space
        # Center lipid logits so they act like a correction
        # rather than an intercept replacement
        lipid_logit_center = lipid_oof_logit_outer_train.mean()
        combined_logit_outer_test = base_logit_outer_test + (
            lipid_logit_outer_test - lipid_logit_center
        )
        oof_probability_additive[test_indices] = _sigmoid(combined_logit_outer_test)

    results = {
        "n": int(len(df)),
        "n_pos": int(y_binary.sum()),
        "base_AP": float(average_precision_score(y_binary, oof_probability_base)),
        "base_ROC_AUC": float(roc_auc_score(y_binary, oof_probability_base)),
        "additive_AP": float(
            average_precision_score(y_binary, oof_probability_additive)
        ),
        "additive_ROC_AUC": float(roc_auc_score(y_binary, oof_probability_additive)),
        "dAP_additive_minus_base": float(
            average_precision_score(y_binary, oof_probability_additive)
            - average_precision_score(y_binary, oof_probability_base)
        ),
        "oof_prob_base": oof_probability_base,
        "oof_prob_additive": oof_probability_additive,
    }
    return results
