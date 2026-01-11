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


def _residualize_train_test(
    lipid_train: np.ndarray,
    lipid_test: np.ndarray,
    cov_train: np.ndarray,
    cov_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # add intercept
    X_train = np.column_stack([np.ones(len(cov_train)), cov_train])
    X_test = np.column_stack([np.ones(len(cov_test)), cov_test])

    # fit OLS on train, apply to train+test
    beta, *_ = np.linalg.lstsq(X_train, lipid_train, rcond=None)

    lipid_train_resid = lipid_train - X_train @ beta
    lipid_test_resid = lipid_test - X_test @ beta
    return lipid_train_resid, lipid_test_resid


def additive_logit_lipid_correction_residualized(
    df: pd.DataFrame,
    random_state: int = 42,
    outer_splits: int = 5,
    inner_splits: int = 5,
):
    df = df.copy()

    covariates = ["age", "sex", "smoker", "duration_illness"]
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

    required = ["true_label"] + covariates + prs_features
    keep = df[required].notna().all(axis=1)
    keep &= ~df[lipid_class_features].isna().all(axis=1)
    df = df.loc[keep].reset_index(drop=True)

    y = (df["true_label"] == 5).astype(int).to_numpy()

    base_feature_columns = covariates + prs_features

    base_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegressionCV(
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
                ),
            ),
        ]
    )

    lipid_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegressionCV(
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
                ),
            ),
        ]
    )

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )

    oof_prob_base = np.zeros(len(df), dtype=float)
    oof_prob_additive = np.zeros(len(df), dtype=float)

    for train_idx, test_idx in outer_cv.split(df, y):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        y_train = y[train_idx]

        X_base_train = train_df[base_feature_columns]
        X_base_test = test_df[base_feature_columns]

        # base OOF on outer-train
        base_oof_prob_train = cross_val_predict(
            clone(base_pipeline),
            X_base_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
        )[:, 1]
        base_oof_logit_train = _safe_logit(base_oof_prob_train)  # noqa: F841

        # base predict on outer-test
        fitted_base = clone(base_pipeline).fit(X_base_train, y_train)
        base_prob_test = fitted_base.predict_proba(X_base_test)[:, 1]
        base_logit_test = _safe_logit(base_prob_test)
        oof_prob_base[test_idx] = base_prob_test

        # residualize lipid class scores on covariates (fit on train, apply to test)
        cov_train = train_df[covariates].to_numpy()
        cov_test = test_df[covariates].to_numpy()

        lipid_train_raw = train_df[lipid_class_features].to_numpy()
        lipid_test_raw = test_df[lipid_class_features].to_numpy()

        lipid_train_resid, lipid_test_resid = _residualize_train_test(
            lipid_train=lipid_train_raw,
            lipid_test=lipid_test_raw,
            cov_train=cov_train,
            cov_test=cov_test,
        )

        # lipid OOF on outer-train (using residualized lipids)
        lipid_oof_prob_train = cross_val_predict(
            clone(lipid_pipeline),
            lipid_train_resid,
            y_train,
            cv=inner_cv,
            method="predict_proba",
        )[:, 1]
        lipid_oof_logit_train = _safe_logit(lipid_oof_prob_train)

        # lipid predict on outer-test (residualized)
        fitted_lipid = clone(lipid_pipeline).fit(lipid_train_resid, y_train)
        lipid_prob_test = fitted_lipid.predict_proba(lipid_test_resid)[:, 1]
        lipid_logit_test = _safe_logit(lipid_prob_test)

        # combine in logit space (center lipid correction on train)
        lipid_center = lipid_oof_logit_train.mean()
        combined_logit_test = base_logit_test + (lipid_logit_test - lipid_center)
        oof_prob_additive[test_idx] = _sigmoid(combined_logit_test)

    results = {
        "n": int(len(df)),
        "n_pos": int(y.sum()),
        "base_AP": float(average_precision_score(y, oof_prob_base)),
        "base_ROC_AUC": float(roc_auc_score(y, oof_prob_base)),
        "additive_resid_AP": float(average_precision_score(y, oof_prob_additive)),
        "additive_resid_ROC_AUC": float(roc_auc_score(y, oof_prob_additive)),
        "dAP_additive_resid_minus_base": float(
            average_precision_score(y, oof_prob_additive)
            - average_precision_score(y, oof_prob_base)
        ),
        "oof_prob_base": oof_prob_base,
        "oof_prob_additive_resid": oof_prob_additive,
    }
    return results
