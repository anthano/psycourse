import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prep_data_for_integrated_analysis(
    multimodal_lipid_subset_df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.25,
    residualize: bool = True,
) -> dict:
    df = multimodal_lipid_subset_df.copy()
    y = (df["true_label"] == 5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    lipid_train = _prep_lipid_data(X_train)
    lipid_test = _prep_lipid_data(X_test)

    lipid_class_train = _prep_lipid_class_data(X_train)
    lipid_class_test = _prep_lipid_class_data(X_test)

    prs_train = _prep_prs_data(X_train)
    prs_test = _prep_prs_data(X_test)

    prs_covar_cols = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]
    lipid_covar_cols = ["age", "sex", "bmi", "duration_illness", "smoker"]

    prs_cov_train = X_train[prs_covar_cols].copy()
    prs_cov_test = X_test[prs_covar_cols].copy()

    lipid_cov_train = X_train[lipid_covar_cols].copy()
    lipid_cov_test = X_test[lipid_covar_cols].copy()

    # impute covariates + features using train only

    prs_cov_train, prs_cov_test = _impute_train_test(prs_cov_train, prs_cov_test)
    lipid_cov_train, lipid_cov_test = _impute_train_test(
        lipid_cov_train, lipid_cov_test
    )
    prs_train, prs_test = _impute_train_test(prs_train, prs_test)
    lipid_train, lipid_test = _impute_train_test(lipid_train, lipid_test)
    lipid_class_train, lipid_class_test = _impute_train_test(
        lipid_class_train, lipid_class_test
    )

    # residualize (fit on train covariates, apply to test)
    prs_train, prs_test = _residualize_block(
        prs_train, prs_test, prs_cov_train, prs_cov_test
    )
    lipid_train, lipid_test = _residualize_block(
        lipid_train, lipid_test, lipid_cov_train, lipid_cov_test
    )
    lipid_class_train, lipid_class_test = _residualize_block(
        lipid_class_train, lipid_class_test, lipid_cov_train, lipid_cov_test
    )

    return {
        "lipid_train": lipid_train,
        "lipid_test": lipid_test,
        "lipid_class_train": lipid_class_train,
        "lipid_class_test": lipid_class_test,
        "prs_train": prs_train,
        "prs_test": prs_test,
        "y_train": y_train.loc[X_train.index],
        "y_test": y_test.loc[X_test.index],
        "prs_cov_train": prs_cov_train,
        "prs_cov_test": prs_cov_test,
        "lipid_cov_train": lipid_cov_train,
        "lipid_cov_test": lipid_cov_test,
    }


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _impute_train_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values using TRAIN stats only:
      - numeric: median
      - non-numeric: mode (most frequent)
    Also replaces +/-inf with NaN before imputing.
    """
    train = train_df.replace([np.inf, -np.inf], np.nan).copy()
    test = test_df.replace([np.inf, -np.inf], np.nan).copy()

    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            fill = train[col].median()
            if pd.isna(fill):
                fill = 0.0
        else:
            mode = train[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else ""
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)

    return train, test


def _residualize_block(
    block_train: pd.DataFrame,
    block_test: pd.DataFrame,
    cov_train: pd.DataFrame,
    cov_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # one-hot encode categoricals (sex, smoker etc.), fit on train, align test
    C_train = pd.get_dummies(cov_train, drop_first=False)
    C_test = pd.get_dummies(cov_test, drop_first=False).reindex(
        columns=C_train.columns, fill_value=0
    )

    # add intercept
    C_train = C_train.copy()
    C_test = C_test.copy()
    C_train.insert(0, "Intercept", 1.0)
    C_test.insert(0, "Intercept", 1.0)

    Xtr = C_train.to_numpy(dtype=float)
    Xte = C_test.to_numpy(dtype=float)

    Ytr = block_train.to_numpy(dtype=float)
    Yte = block_test.to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)

    resid_tr = Ytr - Xtr @ beta
    resid_te = Yte - Xte @ beta

    resid_train_df = pd.DataFrame(
        resid_tr, index=block_train.index, columns=block_train.columns
    )
    resid_test_df = pd.DataFrame(
        resid_te, index=block_test.index, columns=block_test.columns
    )

    return resid_train_df, resid_test_df


def _prep_lipid_data(df: pd.DataFrame) -> pd.DataFrame:
    lipid_cols = [col for col in df.columns if "gpeak" in col]
    return df[lipid_cols].copy()


def _prep_lipid_class_data(df: pd.DataFrame) -> pd.DataFrame:
    lipid_class_cols = [
        "LPE",
        "PC",
        "PC_O",
        "PC_P",
        "PE",
        "PE_P",
        "TAG",
        "dCer",
        "dSM",
        "CAR",
        "CE",
        "DAG",
        "FA",
        "LPC",
        "LPC_O",
        "LPC_P",
    ]
    lipid_class_cols = [col for col in lipid_class_cols if col in df.columns]
    return df[lipid_class_cols].copy()


def _prep_prs_data(df: pd.DataFrame) -> pd.DataFrame:
    prs_cols = [col for col in df.columns if "PRS" in col]
    return df[prs_cols].copy()
