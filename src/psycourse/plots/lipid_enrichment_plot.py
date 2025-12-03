import matplotlib.pyplot as plt
import numpy as np


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
    variant,
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
    classes_in_data = merged["class"].unique().tolist()
    classes = classes_in_data

    # list of coefficient arrays per class
    data = [merged.loc[merged["class"] == cl, "coef"].values for cl in classes]

    # figure
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, positions=np.arange(len(classes)), manage_ticks=False)  # noqa:F841

    ax.axhline(0, linestyle="--")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_ylabel("Lipid coefficient (probability of class 5)")
    ax.set_xlabel(f"Lipid class, covariate: {variant}")

    # mark significant classes (GSEA FDR < threshold) with an asterisk
    sig_classes = set(enrich_df.index[enrich_df["FDR"] < fdr_thresh])
    ymin, ymax = ax.get_ylim()
    y_text = ymax + 0.02 * (ymax - ymin)
    for i, cl in enumerate(classes):
        if cl in sig_classes:
            ax.text(i, y_text, "*", ha="center", va="bottom")

    ax.set_ylim(ymin, y_text + 0.05 * (ymax - ymin))
    fig.tight_layout()
    return fig, ax
