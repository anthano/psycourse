import pandas as pd
import pingouin as pg
import statsmodels.api as sm


def mediation_analysis(multimodal_lipid_subset_df, lipid_enrichment_result_df):
    """Mediation analysis to assess whether lipids mediate the effect of PRS
    on severe psychosis cluster.

    The key change from the original: M (lipid) is no longer pre-residualized.
    Instead, Zm_cols are passed as covariates directly to pg.mediation_analysis,
    so that the b-path (M → Y) and c'-path (X → Y) are both properly adjusted
    for confounders of M and Y. X (PRS) is still pre-residualized against genetic
    PCs and demographic variables that are not valid mediator covariates.

    Args:
        multimodal_lipid_subset_df (pd.DataFrame): DataFrame with all needed columns.
        lipid_enrichment_result_df (pd.DataFrame): lipid enrichment results.

    Returns:
        pd.DataFrame: Tidy mediation results.
    """
    lipid_enrichment_result_df = lipid_enrichment_result_df.copy()
    lipid_cols = _get_lipid_class_cols(lipid_enrichment_result_df)
    prs_cols = ["BD_PRS", "SCZ_PRS", "Education_PRS"]

    # Zx: confounders of X (PRS) — genetic PCs + basic demographics.
    # These are residualized out of X before mediation because they are
    # upstream of X and not appropriate to include as covariates in the
    # M ~ X and Y ~ X, M regressions (they are not confounders of M→Y).
    Zx_cols = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5"]

    # Zm: confounders of M (lipid) and Y (psychosis probability).
    # These are passed directly to pingouin so they adjust both the
    # b-path and c'-path properly.
    Zm_cols = ["age", "sex", "bmi", "duration_illness", "smoker"]

    df = _prep_data(multimodal_lipid_subset_df, prs_cols, lipid_cols, Zx_cols + Zm_cols)

    all_lipid_results_per_prs = {}
    for prs in prs_cols:
        per_lipid = {}
        for lipid in lipid_cols:
            needed = list(
                dict.fromkeys([prs, lipid, "prob_class_5"] + Zx_cols + Zm_cols)
            )
            d = df[needed].dropna().copy()

            # Residualize X against Zx only (removes genetic + demographic
            # confounding from the PRS before it enters the mediation model).
            x_col = f"{prs}_resid"
            d[x_col] = residualize(d[prs], d[Zx_cols])

            # Drop rows where residualization failed (edge case with too few obs)
            d = d.dropna(subset=[x_col])

            # Run mediation with raw lipid as M and Zm as covariates.
            # pingouin fits:
            #   M ~ X + covariates          → a-path
            #   Y ~ X + M + covariates      → b-path, c'-path
            #   Y ~ X + covariates          → total effect (c-path)
            # Indirect = a * b, tested via bootstrap CIs.
            per_lipid[lipid] = pg.mediation_analysis(
                data=d,
                x=x_col,
                m=lipid,
                y="prob_class_5",
                covar=Zm_cols,
                n_boot=5000,
                alpha=0.05,
                seed=42,
            )

        all_lipid_results_per_prs[prs] = per_lipid

    return _mediation_dict_to_tidy(all_lipid_results_per_prs)


########################################################################################
# HELPER FUNCTIONS
########################################################################################


def _get_lipid_class_cols(lipid_enrichment_result_df):
    """Return lipid class columns that passed FDR < 0.05 in enrichment analysis."""
    return [
        col
        for col in lipid_enrichment_result_df.index
        if lipid_enrichment_result_df.loc[col, "FDR"] < 0.05
    ]


def _prep_data(multimodal_lipid_subset_df, prs_cols, lipid_cols, covar_cols):
    """Select and encode columns needed for analysis."""
    all_cols = list(set(prs_cols + lipid_cols + covar_cols + ["prob_class_5"]))
    df = multimodal_lipid_subset_df[all_cols].copy()
    df["sex"] = df["sex"].map({"F": 0, "M": 1}).astype(float)
    df["smoker"] = df["smoker"].map({"never": 0, "former": 1, "yes": 2}).astype(float)
    return df


def _mediation_dict_to_tidy(all_lipid_results_per_prs: dict) -> pd.DataFrame:
    """Flatten nested dict of pingouin mediation results into a tidy DataFrame."""
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
    """Return residuals of s after regressing out Z (OLS), preserving original index."""
    tmp = pd.concat([s, Z], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    y = tmp.iloc[:, 0].astype(float)
    X = sm.add_constant(tmp.iloc[:, 1:].astype(float), has_constant="add")
    fit = sm.OLS(y, X).fit()
    return (y - fit.fittedvalues).reindex(s.index)
