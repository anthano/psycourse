import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def severe_cluster5_classification(multimodal_df, random_state=42):
    """
    Classification of severe psychosis subtype based on PRS and lipidomic data.

    Args:
    multimodal_df (pd.DataFrame): Dataframe containing all data.

    Returns:

    """

    df = multimodal_df.copy()

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

    feature_sets = {
        "cov": covariates,
        "cov_prs": covariates + prs_features,
        "cov_lip": covariates + lipid_features,
        "full": covariates + prs_features + lipid_features,
    }

    y = (df["true_label"] == 5).astype(int).to_numpy()
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    hyperparameter_space = np.logspace(-4, 3, 15)
    l1_ratios = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
    # n_pcs = 10

    results = {}

    for name, features in feature_sets.items():
        X = df[features].to_numpy()

        clf_cv = LogisticRegressionCV(
            Cs=hyperparameter_space,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=l1_ratios,
            class_weight="balanced",
            cv=3,
            max_iter=5000,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )

        uses_lipids = any(f in lipid_features for f in features)

        if uses_lipids:
            X = df[features]

            cov_block = [cov for cov in covariates if cov in features]
            prs_block = [prs for prs in prs_features if prs in features]
            lipids_block = [lipid for lipid in lipid_features if lipid in features]

            preprocessing = ColumnTransformer(
                transformers=[
                    (
                        "cov_prs",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        cov_block + prs_block,
                    ),
                    (
                        "lipids",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                                # (
                                #    "pca",
                                #    PCA(n_components=n_pcs, random_state=random_state),
                                # ),
                            ]
                        ),
                        lipids_block,
                    ),
                ],
                remainder="drop",
            )

            model = Pipeline([("preprocessing", preprocessing), ("classifier", clf_cv)])
        else:
            X = df[features].to_numpy()
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", clf_cv),
                ]
            )

        oof_prob = cross_val_predict(model, X, y, cv=outer_cv, method="predict_proba")[
            :, 1
        ]

        results[name] = {
            "n": int(len(df)),
            "n_pos": int(y.sum()),
            "oof_average_precision": float(average_precision_score(y, oof_prob)),
            "oof_ROC_AUC": float(roc_auc_score(y, oof_prob)),
            "oof_prob": oof_prob,
        }

    results["dAP_PRS"] = (
        results["cov_prs"]["oof_average_precision"]
        - results["cov"]["oof_average_precision"]
    )
    results["dAP_Lip"] = (
        results["cov_lip"]["oof_average_precision"]
        - results["cov"]["oof_average_precision"]
    )
    results["dAP_Lip_givenPRS"] = (
        results["full"]["oof_average_precision"]
        - results["cov_prs"]["oof_average_precision"]
    )
    results["dAP_PRS_givenLip"] = (
        results["full"]["oof_average_precision"]
        - results["cov_lip"]["oof_average_precision"]
    )

    return results
