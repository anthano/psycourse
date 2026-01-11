import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stacked_cluster5_model(
    df: pd.DataFrame,
    random_state: int = 42,
    outer_splits: int = 5,
    inner_splits: int = 5,
):
    df = df.copy()

    lipid_features = [
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
    df["sex"] = df["sex"].map({"F": 0, "M": 1})
    df["smoker"] = df["smoker"].map({"former": 0, "never": 1, "yes": 2})

    base_cols = ["true_label"] + covariates + prs_features
    keep = df[base_cols].notna().all(axis=1)
    if lipid_features:
        keep &= ~df[lipid_features].isna().all(axis=1)

    df = df.loc[keep].reset_index(drop=True)

    y = (df["true_label"] == 5).astype(int).to_numpy()

    # feature blocks
    base_feats = covariates + prs_features
    lipid_class_features = lipid_features

    # base learner (cov+PRS)
    base_clf = LogisticRegressionCV(
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
    base_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", base_clf),
        ]
    )

    # meta learner (keep it stable)
    meta_clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    meta_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", meta_clf),
        ]
    )

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv_for_oof = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )

    oof_prob_stacked = np.zeros(len(df), dtype=float)
    oof_prob_base = np.zeros(len(df), dtype=float)

    lipid_class_features = lipid_features

    for train_indices, test_indices in outer_cv.split(df, y):
        train_data_outer = df.iloc[train_indices]
        test_data_outer = df.iloc[test_indices]

        y_train_outer = y[train_indices]

        X_base_train_outer = train_data_outer[base_feats]
        X_base_test_outer = test_data_outer[base_feats]

        # 1) OOF base preds on outer-train
        base_oof_prob_outer_train = cross_val_predict(
            clone(base_model),
            X_base_train_outer,
            y_train_outer,
            cv=inner_cv_for_oof,
            method="predict_proba",
        )[:, 1]

        # 2) Base fit on outer-train, predict outer-test
        fitted_base_outer_train = clone(base_model).fit(
            X_base_train_outer, y_train_outer
        )
        base_prob_outer_test = fitted_base_outer_train.predict_proba(X_base_test_outer)[
            :, 1
        ]
        oof_prob_base[test_indices] = base_prob_outer_test

        # --- HARD ASSERTS (fail early) ---
        assert len(base_oof_prob_outer_train) == len(train_data_outer)
        assert len(base_prob_outer_test) == len(test_data_outer)

        # 3) Build meta features as NUMPY (no index alignment)
        lipid_train_outer = train_data_outer[lipid_class_features].to_numpy()
        lipid_test_outer = test_data_outer[lipid_class_features].to_numpy()

        X_meta_train_outer = np.column_stack(
            [base_oof_prob_outer_train, lipid_train_outer]
        )
        X_meta_test_outer = np.column_stack([base_prob_outer_test, lipid_test_outer])

        # 4) Fit meta and predict
        fitted_meta_outer_train = clone(meta_model).fit(
            X_meta_train_outer, y_train_outer
        )
        oof_prob_stacked[test_indices] = fitted_meta_outer_train.predict_proba(
            X_meta_test_outer
        )[:, 1]

    results = {
        "n": int(len(df)),
        "n_pos": int(y.sum()),
        "base_AP": float(average_precision_score(y, oof_prob_base)),
        "base_ROC_AUC": float(roc_auc_score(y, oof_prob_base)),
        "stacked_AP": float(average_precision_score(y, oof_prob_stacked)),
        "stacked_ROC_AUC": float(roc_auc_score(y, oof_prob_stacked)),
        "dAP_stacked_minus_base": float(
            average_precision_score(y, oof_prob_stacked)
            - average_precision_score(y, oof_prob_base)
        ),
        "oof_prob_base": oof_prob_base,
        "oof_prob_stacked": oof_prob_stacked,
    }
    return results
