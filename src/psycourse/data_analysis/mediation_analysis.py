import pandas as pd
import pingouin as pg
import statsmodels.api as sm


def mediation_analysis(multimodal_lipid_subset_df, lipid_enrichment_result_df):
    """Mediation analysis to assess whether much lipids mediate the effect of PRS
     on severe psychosis cluster.
     Args:
        multimodal_lipid_subset_df (pd.DataFrame): DataFrame with all needed columns.
        lipid_enrichment_result_df (pd.DataFrame): lipid enrichment results.

    Returns:
        mediation_results (dict): Dictionary with mediation analysis results.
    """
    lipid_enrichment_result_df = lipid_enrichment_result_df.copy()
    lipid_cols = _get_lipid_class_cols(lipid_enrichment_result_df)
    prs_cols = [
        "BD_PRS",
        "SCZ_PRS",
        "MDD_PRS",
        "Lipid_BD_PRS",
        "Lipid_SCZ_PRS",
        "Lipid_MDD_PRS",
    ]
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

    df = _prep_data(multimodal_lipid_subset_df, prs_cols, lipid_cols, covars)
    Zx_cols = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]
    Zm_cols = ["age", "sex", "bmi", "duration_illness", "smoker"]

    all_lipid_results_per_prs = {}
    for prs in prs_cols:
        per_lipid = {}
        for lipid in lipid_cols:
            d = df[[prs, lipid, "prob_class_5", *Zx_cols, *Zm_cols]].copy()

            x_col = f"{prs}_resid"
            m_col = f"{lipid}_resid"
            d[x_col] = residualize(d[prs], d[Zx_cols])
            d[m_col] = residualize(d[lipid], d[Zm_cols])

            d2 = d[[x_col, m_col, "prob_class_5"]].dropna()
            per_lipid[lipid] = pg.mediation_analysis(
                d2,
                x=x_col,
                m=m_col,
                y="prob_class_5",
                covar=None,
                n_boot=5000,
                alpha=0.05,
            )

        all_lipid_results_per_prs[prs] = per_lipid

    return _mediation_dict_to_tidy(all_lipid_results_per_prs)


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _get_lipid_class_cols(lipid_enrichment_result_df):
    lipid_class_cols = [
        col
        for col in lipid_enrichment_result_df.index
        if lipid_enrichment_result_df.loc[col, "FDR"] < 0.05
    ]
    return lipid_class_cols


def _prep_data(multimodal_lipid_subset_df, prs_cols, lipid_cols, covars):
    df = multimodal_lipid_subset_df[
        prs_cols + lipid_cols + covars + ["prob_class_5"]
    ].copy()

    df["sex"] = df["sex"].map({"F": 0, "M": 1}).astype(float)
    df["smoker"] = df["smoker"].map({"never": 0, "former": 1, "yes": 2}).astype(float)

    return df


def _mediation_dict_to_tidy(all_lipid_results_per_prs: dict) -> pd.DataFrame:
    rows = []
    for prs, lipid_map in all_lipid_results_per_prs.items():
        for lipid, res in lipid_map.items():
            r = res.copy()
            if "path" not in r.columns:
                r = r.reset_index().rename(columns={"index": "path"})
            r.insert(0, "prs", prs)
            r.insert(1, "mediator", lipid)
            rows.append(r)
    return (
        pd.concat(rows, ignore_index=True)
        .set_index(["prs", "mediator", "path"])
        .sort_index()
    )


def residualize(s: pd.Series, Z: pd.DataFrame) -> pd.Series:
    tmp = pd.concat([s, Z], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    y = tmp.iloc[:, 0].astype(float)
    X = sm.add_constant(tmp.iloc[:, 1:].astype(float), has_constant="add")
    fit = sm.OLS(y, X).fit()
    return (y - fit.fittedvalues).reindex(s.index)
