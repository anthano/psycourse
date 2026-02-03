import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def enrichment_strength_plot(enrichment_results, variant):
    """Plot enrichment strength for lipid classes based on GSEA results.
    Args:
        enrichment_results (pd.DataFrame): DataFrame containing GSEA results
        with columns 'NES' and 'FDR'.
        variant (str): Name of the variant for labeling purposes via pytask loop.

    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """
    result = enrichment_results.copy()  # output of lipid_class_enrichment_gsea
    # order by Normalized Enrichment Score
    result = result.sort_values("NES")
    fig, ax = plt.subplots(figsize=(6, 4))
    y = range(len(result))
    ax.hlines(y, 0, result["NES"])
    ax.plot(result["NES"], y, "o")
    ax.axvline(0, linestyle="--")

    # mark significant classes
    sig_mask = result["FDR"] < 0.05
    ax.plot(
        result["NES"][sig_mask],
        [y_i for y_i, m in zip(y, sig_mask, strict=False) if m],
        "o",
    )  # change marker style/edgecolor to highlight

    ax.set_yticks(y)
    ax.set_yticklabels(result.index)
    ax.set_xlabel(f"Normalized Enrichment Score (NES), covariate: {variant}")
    ax.set_title("Lipid class GSEA (NES)")

    plt.tight_layout()

    return fig, ax


def plot_lipid_coef_distributions(
    results_df,
    annot_df,
    enrich_df,
    coef_col="coef",
    class_col="class",
    fdr_thresh=0.05,
    figsize=(8, 4),
):
    """
    Boxplots of lipid coefficients per class, marking classes
    that are significantly enriched in GSEA (FDR < fdr_thresh).

    Args:
        results_df (pd.DataFrame): DataFrame with lipid regression results.
        annot_df (pd.DataFrame): DataFrame with lipid class annotations.
        enrich_df (pd.DataFrame): DataFrame with lipid enrichment results.
        variant (str): Covariate/variant name for labeling.

    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """

    annot_use = annot_df[[class_col]].rename(columns={class_col: "class"})
    merged = (
        results_df[[coef_col]]
        .rename(columns={coef_col: "coef"})
        .join(annot_use, how="inner")
        .dropna()
    )
    merged["class"] = merged["class"].astype(str)

    # order classes by NES from enrichment results if available
    tested = set(enrich_df.index)

    classes_in_data = merged["class"].unique().tolist()
    classes = [cl for cl in classes_in_data if cl in tested]

    # list of coefficient arrays per class
    data = [merged.loc[merged["class"] == cl, "coef"].values for cl in classes]

    # figure

    fig, ax = plt.subplots(figsize=figsize)

    # --- colors: highlight significant classes ---
    sig_classes = set(enrich_df.index[enrich_df["FDR"] < fdr_thresh])

    color_sig = "#3B4CC0"
    color_nonsig = "#D9D9D9"
    edge = "#2b2b2b"

    bp = ax.boxplot(
        data,
        positions=np.arange(len(classes)),
        manage_ticks=False,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.4,
            markerfacecolor=edge,
            markeredgecolor=edge,
            markeredgewidth=0,
        ),
    )

    for patch, cl in zip(bp["boxes"], classes, strict=False):
        patch.set_facecolor(color_sig if cl in sig_classes else color_nonsig)
        patch.set_edgecolor(edge)
        patch.set_alpha(0.85)

    for k in ("whiskers", "caps", "medians"):
        for artist in bp[k]:
            artist.set_color(edge)
            artist.set_linewidth(1.2)

    ax.grid(axis="y", alpha=0.25)

    all_vals = np.concatenate([v for v in data if len(v)])
    m = np.max(np.abs(all_vals))
    pad = 0.12 * m
    ax.set_ylim(-(m + pad), (m + pad))

    star_gap = 0.06 * m
    y_top = ax.get_ylim()[1]
    for i, (cl, vals) in enumerate(zip(classes, data, strict=False)):
        if cl in sig_classes and len(vals):
            y_star = min(np.max(vals) + star_gap, y_top - 0.02 * (2 * (m + pad)))
            ax.text(i, y_star, "*", ha="center", va="bottom", fontsize=12)

    # labels etc.
    ax.axhline(0, linestyle="--", color=edge, alpha=0.6)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Lipid coefficient (probability of class 5)")
    ax.set_xlabel("Lipid class")
    fig.tight_layout()

    legend_handles = [
        Patch(facecolor=color_sig, edgecolor=edge, alpha=0.85, label="Significant"),
        Patch(
            facecolor=color_nonsig, edgecolor=edge, alpha=0.85, label="Not significant"
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)

    return fig, ax
