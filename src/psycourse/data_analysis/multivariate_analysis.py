import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def multivariate_regression(multimodal_df):
    """Multivariate regression analysis to assess variance of severe psychosis cluster
    explained by lipid species and lipid-psychiatric PRS.
    Args:
        multimodal_df (pd.DataFrame): DataFrame with all needed columns.

    Returns:
        multivariate_regression_results (dict): Dictionary with R² results."""

    df = multimodal_df.copy().dropna()
    df["sex"] = df["sex"].map({"F": 0, "M": 1}).astype(np.int8)
    df["smoker"] = df["smoker"].map({"former": 0, "never": 1, "yes": 2}).astype(np.int8)

    target_col = "prob_class_5"
    covariates = [
        "age",
        "bmi",
        "sex",
        "smoker",
        "duration_illness",
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pc5",
    ]
    prs_features = ["SCZ_PRS", "BD_PRS", "MDD_PRS"]
    lipid_features = [col for col in df.columns if col.startswith("gpeak")]
    # Remove rows where all lipid features are NaN
    data_with_lipids = df[~df[lipid_features].isna().all(axis=1)]
    relevant_cols = covariates + lipid_features + prs_features + [target_col]

    analysis_data = data_with_lipids[relevant_cols].copy()  # noqa: F841

    y = df[target_col].values
    X_cov = df[covariates].values
    X_covPRS = df[covariates + prs_features].values
    X_covLip = df[covariates + lipid_features].values
    X_full = df[covariates + prs_features + lipid_features].values

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    R2_cov = _cv_r2(X_cov, y, cv)
    R2_covPRS = _cv_r2(X_covPRS, y, cv)
    R2_covLip = _cv_r2(X_covLip, y, cv)
    R2_full = _cv_r2(X_full, y, cv)

    dR2_PRS = R2_covPRS - R2_cov
    dR2_Lip = R2_covLip - R2_cov
    dR2_Lip_givenPRS = R2_full - R2_covPRS
    dR2_PRS_givenLip = R2_full - R2_covLip

    multivariate_regression_results = {
        "R2_cov": R2_cov,
        "R2_covPRS": R2_covPRS,
        "R2_covLip": R2_covLip,
        "R2_full": R2_full,
        "dR2_PRS": dR2_PRS,
        "dR2_Lip": dR2_Lip,
        "dR2_Lip_givenPRS": dR2_Lip_givenPRS,
        "dR2_PRS_givenLip": dR2_PRS_givenLip,
    }

    print(df["prob_class_5"].describe())
    print("Var(y):", df["prob_class_5"].var())

    return multivariate_regression_results


def _cv_r2(X, y, cv):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0]))
    y_pred = cross_val_predict(model, X, y, cv=cv)
    return r2_score(y, y_pred)
