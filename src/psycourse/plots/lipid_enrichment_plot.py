import matplotlib.pyplot as plt


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
