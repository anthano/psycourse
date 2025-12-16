import pandas as pd
import pingouin as pg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def mediation_analysis(multimodal_df, prs):
    """Mediation analysis to assess whether much lipids mediate the effect of PRS
     on severe psychosis cluster.
     Args:
        multimodal_df (pd.DataFrame): DataFrame with all needed columns.
        prs (str): Polygenic risk score to analyze.
        One of 'SCZ_PRS', 'BD_PRS','MDD_PRS'.

    Returns:
        mediation_results (dict): Dictionary with mediation analysis results.
    """

    df = _prep_data(multimodal_df, prs)
    lipid_cols = [col for col in df.columns if col.startswith("gpeak")]

    df_with_pcs = _perform_pca(df, lipid_cols, n_components=5, random_state=42)

    for col in ["prob_class_5", "lipid_PC1"]:
        df_with_pcs[col + "_z"] = (
            df_with_pcs[col] - df_with_pcs[col].mean()
        ) / df_with_pcs[col].std(ddof=0)

    covars = [
        "age",
        "sex",
        "bmi",
        "smoker",
        "duration_illness",
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pc5",
    ]

    analysis_cols = [prs, "prob_class_5_z", "lipid_PC1_z"] + covars
    df_med_analysis = df_with_pcs[analysis_cols].dropna().copy()

    df_med_analysis = df_med_analysis.apply(pd.to_numeric, errors="coerce")
    df_med_analysis = df_med_analysis.dropna().astype("float64")

    med_res = pg.mediation_analysis(
        data=df_med_analysis,
        x=f"{prs}",
        m="lipid_PC1_z",
        y="prob_class_5_z",
        covar=covars,
        n_boot=5000,
        alpha=0.05,
    )

    return med_res


def _prep_data(multimodal_df, prs):
    lipid_features = [col for col in multimodal_df.columns if col.startswith("gpeak")]

    base_cols = [
        "age",
        "sex",
        "bmi",
        "smoker",
        "duration_illness",
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pc5",
        "prob_class_5",
        prs,
    ]

    df = multimodal_df[base_cols + lipid_features].copy()

    df["sex"] = df["sex"].map({"F": 0, "M": 1}).astype(float)
    df["smoker"] = df["smoker"].map({"never": 0, "former": 1, "yes": 2}).astype(float)

    has_any_lipid = df[lipid_features].notna().any(axis=1)
    df = df.loc[has_any_lipid].copy()

    return df


def _perform_pca(df, lipid_cols, n_components=5, random_state=42):
    df_with_pcs = df.copy()
    lipid_df = df[lipid_cols].copy()
    lipid_df = lipid_df.fillna(lipid_df.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(lipid_df.values)

    pca = PCA(n_components=n_components, random_state=random_state)
    principal_components = pca.fit_transform(X_scaled)

    for i in range(n_components):
        df_with_pcs[f"lipid_PC{i+1}"] = principal_components[:, i]

    return df_with_pcs
