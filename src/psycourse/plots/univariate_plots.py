import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from psycourse.config import (
    PLOT_CAPSIZE,
    PLOT_CAPTHICK,
    PLOT_DOT_POINT_SIZE,
    PLOT_ECOLOR,
    PLOT_EDGECOLOR,
    PLOT_ELINEWIDTH,
    PLOT_FOREST_POINT_SIZE,
    PLOT_LIP_FDR,
    PLOT_LIP_NOM,
    PLOT_LIP_NS,
    PLOT_PRS_FDR,
    PLOT_PRS_NOM,
    PLOT_PRS_NS,
    PLOT_RCPARAMS,
)


def _palette_colors(palette: str) -> tuple[str, str, str]:
    """Return (fdr_color, nom_color, ns_color) for the requested palette."""
    if palette == "lipid":
        return PLOT_LIP_FDR, PLOT_LIP_NOM, PLOT_LIP_NS
    return PLOT_PRS_FDR, PLOT_PRS_NOM, PLOT_PRS_NS


def _sig_colors(pval_arr, fdr_arr, palette: str = "prs"):
    """Return per-row tier color based on p-value and FDR arrays."""
    fdr_c, nom_c, ns_c = _palette_colors(palette)
    colors = []
    for p, q in zip(pval_arr, fdr_arr, strict=False):
        if q < 0.05:
            colors.append(fdr_c)
        elif p < 0.05:
            colors.append(nom_c)
        else:
            colors.append(ns_c)
    return colors


def _legend_elements(palette: str = "prs"):
    """Return fresh legend handles for the three significance tiers."""
    fdr_c, nom_c, ns_c = _palette_colors(palette)
    return [
        Patch(facecolor=fdr_c, edgecolor="k", label="FDR < 0.05"),
        Patch(facecolor=nom_c, edgecolor="k", label="p < 0.05"),
        Patch(facecolor=ns_c, edgecolor="k", label="n.s."),
    ]


def _format_prs_labels(labels):
    """Apply standard PRS label formatting."""
    renamed = []
    for lbl in labels:
        new_lbl = lbl.replace("_", "-")
        new_lbl = new_lbl.replace("Education-PRS", "EA-PRS")
        new_lbl = new_lbl.replace("Lipid-Edu-PRS", "Lipid-EA-PRS")
        renamed.append(new_lbl)
    return renamed


############# Lipids ###################


def plot_univariate_lipid_regression(
    lipid_results_top20, cleaned_annotation_df, ax=None
):
    """
    Plot lipid GLM coefficients with 95% CIs.
    Points and error bars colored by significance tier.
    """
    mapping = cleaned_annotation_df["lipid_species"].to_dict()
    lipid_results_top20 = lipid_results_top20.rename(index=mapping)
    lipid_sorted = lipid_results_top20.sort_values("FDR").copy()
    y = np.arange(len(lipid_sorted))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    sns.set_theme(style="whitegrid")

    colors = _sig_colors(
        lipid_sorted["pval"].to_numpy(),
        lipid_sorted["FDR"].to_numpy(),
        palette="lipid",
    )

    # Per-point colored error bars (95% CI) — whiskers always black
    for i, (_, row) in enumerate(lipid_sorted.iterrows()):
        ax.errorbar(
            row["coef"],
            i,
            xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
            fmt="none",
            ecolor=PLOT_ECOLOR,
            elinewidth=PLOT_ELINEWIDTH,
            capsize=PLOT_CAPSIZE,
            capthick=PLOT_CAPTHICK,
            zorder=2,
        )

    # Scatter with tier colors
    ax.scatter(
        lipid_sorted["coef"].to_numpy(),
        y,
        c=colors,
        edgecolor=PLOT_EDGECOLOR,
        s=PLOT_FOREST_POINT_SIZE,
        zorder=3,
    )

    # zero (null) line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    # y labels, most significant at the TOP
    ax.set_yticks(y, lipid_sorted.index)
    ax.invert_yaxis()

    # symmetric x limits for balance
    lim = np.nanmax(np.abs(lipid_sorted[["ci_low", "ci_high", "coef"]].to_numpy()))
    if np.isfinite(lim) and lim > 0:
        ax.set_xlim(-1.1 * lim, 1.1 * lim)

    # cleaner grid
    ax.grid(True, axis="x", color="0.9")  # turn x-grid on
    ax.grid(False, axis="y")  # turn y-grid off
    ax.set_axisbelow(True)
    sns.despine(left=True)

    ax.set_xlabel("Regression Coefficient")
    ax.set_ylabel("Lipids", labelpad=10)

    ax.legend(
        handles=_legend_elements(palette="lipid"), loc="lower right", frameon=True
    )
    if standalone:
        plt.tight_layout()

    return fig, ax


def plot_univariate_lipid_class_regression(lipid_class_results, cleaned_annotation_df):
    """Plot the lipid classes associated with cluster 5 probability using regression
    coefficients and FDR values.
    Args:
        lipid_class_results (pd.DataFrame): DataFrame containing the associations
         between lipid class with columns 'coef' and 'FDR'.
        cleaned_annotation_df (pd.DataFrame): DataFrame containing the cleaned
         lipid annotation data with 'lipid species' column.
    Returns:
        None: Displays a plot of the regression coefficients with FDR as color.
    """

    # Map lipid species to lipid class names for y-axis labels
    mapping = cleaned_annotation_df["lipid_species"].to_dict()
    lipid_class_results = lipid_class_results.rename(index=mapping)

    # Sort by FDR again to fix order
    lipid_class_results_sorted = lipid_class_results.sort_values("FDR")

    colors = _sig_colors(
        lipid_class_results_sorted["pval"].to_numpy(),
        lipid_class_results_sorted["FDR"].to_numpy(),
        palette="lipid",
    )

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Draw lines
    for i, row in lipid_class_results_sorted.iterrows():
        plt.plot([0, row["coef"]], [i, i], "k-", lw=0.5)

    # Draw points
    plt.scatter(
        lipid_class_results_sorted["coef"],
        lipid_class_results_sorted.index,
        c=colors,
        s=PLOT_DOT_POINT_SIZE,
        edgecolor=PLOT_EDGECOLOR,
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Regression Coefficient")
    plt.ylabel("Lipid Class", labelpad=10)
    plt.legend(
        handles=_legend_elements(palette="lipid"), loc="lower right", frameon=True
    )
    plt.tight_layout()
    # plt.show()


def plot_univariate_lipid_extremes(top20):
    """Plot the top 20 lipids associated with cluster 5 probability using regression
    coefficients and FDR values for the top50 vs. bottom top50 cases.
    Args:
        top20 (pd.DataFrame): DataFrame containing the top 20 lipids with columns
        'coef' and 'FDR'.
    Returns:
        None: Displays a plot of the regression coefficients with FDR as color.
    """
    # Sort top 20 by FDR again to fix order
    top20_sorted = top20.sort_values("FDR")

    colors = _sig_colors(
        top20_sorted["pval"].to_numpy(),
        top20_sorted["FDR"].to_numpy(),
        palette="lipid",
    )

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Draw lines
    for i, row in top20_sorted.iterrows():
        plt.plot([0, row["coef"]], [i, i], "k-", lw=0.5)

    # Draw points
    plt.scatter(
        top20_sorted["coef"],
        top20_sorted.index,
        c=colors,
        s=PLOT_DOT_POINT_SIZE,
        edgecolor=PLOT_EDGECOLOR,
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Regression Coefficient")
    plt.ylabel("Lipid", labelpad=10)
    plt.legend(
        handles=_legend_elements(palette="lipid"), loc="lower right", frameon=True
    )
    plt.tight_layout()
    # plt.show()


def plot_corr_matrix_lipid_top20(
    multimodal_df, top20_results_df, cleaned_annotation_df
):
    """Plot a correlation matrix for the top 20 lipids.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing lipidomic data (and more).
        top20_results_df (pd.DataFrame): DataFrame containing top 20 results with
        lipids in index.

    Returns:
        None: Displays a correlation matrix plot.
    """
    top20_df = multimodal_df[multimodal_df.columns.intersection(top20_results_df.index)]
    mapping = cleaned_annotation_df["lipid_species"].to_dict()
    top20_df = top20_df.rename(columns=mapping)
    top20_df_corr_matrix = top20_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        top20_df_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix of Top 20 Lipids")
    plt.tight_layout()
    # plt.show()


def plot_perm_enrichment(
    enrich_df,
    top_n=16,
    title="Lipid class enrichment (permutation-based)",
):
    """
    Dot plot of class enrichment:
      x = ES (mean-rank difference; >0 = enriched toward top of ranking)
      y = lipid class (top_n by FDR)
      color = significance tier (FDR < 0.05 / p < 0.05 / n.s.)
      size = n_in_class
    """
    if enrich_df.empty:
        raise ValueError("enrich_df is empty.")

    df = enrich_df.sort_values("FDR").head(top_n).iloc[::-1].copy()  # most sig at top

    colors = _sig_colors(df["pval"].to_numpy(), df["FDR"].to_numpy(), palette="lipid")

    sns.set_theme(style="whitegrid", rc=PLOT_RCPARAMS)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter
    ax.scatter(
        df["ES"],
        df.index,
        s=20 + 6 * df["n_in_class"],  # size by class size
        c=colors,
        edgecolor="k",
        linewidth=0.6,
        zorder=3,
    )

    # Zero line = no enrichment
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    # Labels / grid
    ax.set_xlabel("Enrichment score (Δ mean rank; >0 = enriched)")
    ax.set_ylabel("Lipid class", labelpad=10)
    ax.grid(True, axis="x", color="0.9")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    sns.despine(left=True)

    # Nice symmetric x-lims with padding
    span = np.nanmax(np.abs(df["ES"]))
    pad = 0.15 * max(0.01, span)
    ax.set_xlim(-span - pad, span + pad)

    ax.legend(
        handles=_legend_elements(palette="lipid"), loc="lower right", frameon=True
    )

    plt.tight_layout()
    return fig, ax


########################################################################################
# PRS
########################################################################################


def plot_univariate_prs_regression(prs_results, ax=None):
    """
    Plot PRS GLM coefficients with 95% CIs.
    Points and error bars colored by significance tier.
    """
    plt.rcParams.update(PLOT_RCPARAMS)

    # Drop Lipid-MDD-PRS
    norm_idx = prs_results.index.str.replace("_", "-")
    prs_results = prs_results[norm_idx != "Lipid-MDD-PRS"]

    prs_sorted = prs_results.sort_values("FDR").copy()
    y = np.arange(len(prs_sorted))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    sns.set_theme(style="whitegrid")

    colors = _sig_colors(prs_sorted["pval"].to_numpy(), prs_sorted["FDR"].to_numpy())

    # Per-point colored error bars (95% CI) — whiskers always black
    for i, (_, row) in enumerate(prs_sorted.iterrows()):
        ax.errorbar(
            row["coef"],
            i,
            xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
            fmt="none",
            ecolor=PLOT_ECOLOR,
            elinewidth=PLOT_ELINEWIDTH,
            capsize=PLOT_CAPSIZE,
            capthick=PLOT_CAPTHICK,
            zorder=2,
        )

    # Scatter with tier colors
    ax.scatter(
        prs_sorted["coef"].to_numpy(),
        y,
        c=colors,
        edgecolor=PLOT_EDGECOLOR,
        s=PLOT_FOREST_POINT_SIZE,
        zorder=3,
    )

    # zero (null) line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    # y labels, most significant at the TOP
    labels = _format_prs_labels(prs_sorted.index.tolist())
    ax.set_yticks(y, labels)
    ax.invert_yaxis()

    # symmetric x limits for balance
    lim = np.nanmax(np.abs(prs_sorted[["ci_low", "ci_high", "coef"]].to_numpy()))
    if np.isfinite(lim) and lim > 0:
        ax.set_xlim(-1.1 * lim, 1.1 * lim)

    # cleaner grid
    ax.grid(True, axis="x", color="0.9")  # turn x-grid on
    ax.grid(False, axis="y")  # turn y-grid off
    ax.set_axisbelow(True)
    sns.despine(left=True)

    ax.set_xlabel("Regression Coefficient")
    ax.set_ylabel("PRS", labelpad=10)

    ax.legend(handles=_legend_elements(palette="prs"), loc="lower right", frameon=True)

    if standalone:
        plt.tight_layout()

    return fig, ax


def plot_univariate_prs_extremes(prs_results):
    # Drop Lipid-MDD-PRS
    norm_idx = prs_results.index.str.replace("_", "-")
    prs_results = prs_results[norm_idx != "Lipid-MDD-PRS"]

    prs_sorted = prs_results.sort_values("FDR")

    colors = _sig_colors(prs_sorted["pval"].to_numpy(), prs_sorted["FDR"].to_numpy())

    labels_formatted = _format_prs_labels(prs_sorted.index.tolist())
    label_map = dict(zip(prs_sorted.index, labels_formatted, strict=False))

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Draw lines
    for idx, row in prs_sorted.iterrows():
        lbl = label_map[idx]
        plt.plot([0, row["coef"]], [lbl, lbl], "k-", lw=0.5)

    # Draw points
    plt.scatter(
        prs_sorted["coef"],
        labels_formatted,
        c=colors,
        s=PLOT_DOT_POINT_SIZE,
        edgecolor=PLOT_EDGECOLOR,
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Regression Coefficient")
    plt.ylabel("PRS", labelpad=10)
    plt.legend(handles=_legend_elements(palette="prs"), loc="lower right", frameon=True)
    plt.tight_layout()
    # plt.show()


def plot_corr_matrix_prs(multimodal_df):
    """Plot a correlation matrix for the top 20 lipids.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing prs (and more).

    Returns:
        None: Displays a correlation matrix plot.
    """
    prs_columns = [col for col in multimodal_df.columns if col.endswith("PRS")]
    prs_df = multimodal_df[prs_columns]
    top20_df_corr_matrix = prs_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        top20_df_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix of PRS")
    plt.tight_layout()
    # plt.show()


def plot_prs_cv_delta_mse(delta_df, title="Out-of-sample error reduction by PRS"):
    """
    Horizontal bar plot of ΔMSE% with plasma-edge colors.
    Yellow (#f0f921) = improvement (>0), Purple (#0d0887) = worse (<=0).
    Error bars match bar colors. Numeric labels are nudged to avoid overlap."""

    plt.rcParams.update(PLOT_RCPARAMS)

    dd = delta_df.sort_values("delta_mse_pct_mean", ascending=True).copy()
    vals = dd["delta_mse_pct_mean"].to_numpy()
    errs = dd["delta_mse_pct_std"].to_numpy()

    # Plasma edge colors
    col_pos = "#fde725"  # warm yellow (high end of plasma)
    col_neg = "#41049d"  # deep purple (low end of plasma)
    bar_colors = [col_pos if v > 0 else col_neg for v in vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Draw bars first (no errorbars here so we can color them per-bar next)
    bars = ax.barh(dd.index, vals, color=bar_colors, edgecolor="black", linewidth=0.6)  # noqa: F841

    # Error bars, semi-transparent version of bar color
    y_pos = np.arange(len(dd))  # noqa: F841
    for i, (v, e) in enumerate(zip(vals, errs, strict=False)):
        if np.isnan(e):
            continue
        ax.errorbar(
            x=v,
            y=i,
            xerr=e,
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            zorder=2,
            alpha=0.5,  # <--- transparency
        )

    # Zero reference
    ax.axvline(0, color="0.5", linestyle="--", linewidth=1)

    # Grid aesthetics
    ax.grid(True, axis="x", color="0.9")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    sns.despine(left=True)

    # Labels / title
    ax.set_xlabel("Δ MSE (%) vs covariates-only (5-fold CV)")
    ax.set_ylabel("PRS", labelpad=10)
    ax.set_title(title)

    # Nice symmetric x-lims with padding
    span = max(0.01, np.nanmax(np.abs(vals) + np.nan_to_num(errs, nan=0)))
    pad = 0.15 * span
    ax.set_xlim(-span - pad, span + pad)

    safe_errs = np.nan_to_num(errs, nan=0.0, posinf=0.0, neginf=0.0)

    # Numeric labels placed outside the furthest edge of bar + error bar
    for i, (v, e) in enumerate(zip(vals, safe_errs, strict=False)):
        txt = f"{abs(v):.2f}%"
        if v >= 0:
            xpos = v + e + 0.02 * span
            ax.text(xpos, i, txt, va="center", ha="left", fontsize=10)
        else:
            xpos = v - e - 0.02 * span
            ax.text(xpos, i, txt, va="center", ha="right", fontsize=10)

    plt.tight_layout()
    return fig, ax
