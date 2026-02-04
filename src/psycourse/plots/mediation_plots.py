import matplotlib.pyplot as plt
import seaborn as sns


def plot_mediation_results(mediation_df):
    """
    Create a coefficient plot for indirect effects from mediation analysis.
    """
    # Extract indirect effects only
    indirect = mediation_df.loc[(slice(None), slice(None), "Indirect"), :].copy()
    indirect = indirect.reset_index()

    # Filter out Lipid_* PRSs
    indirect = indirect[~indirect["prs"].str.startswith("Lipid_")]

    # Define colors that match your theme
    prs_colors = {
        "BD_PRS": "#3B4CC0",  # Your blue
        "SCZ_PRS": "#7B68EE",  # Medium purple (between your blue and plasma)
        "MDD_PRS": "#B83280",  # Magenta-purple (from plasma range)
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique lipids and PRSs
    lipids = indirect["mediator"].unique()
    prss = indirect["prs"].unique()

    # Create y-positions for each lipid
    y_pos = {lipid: i for i, lipid in enumerate(lipids)}
    offset_width = 0.25
    prs_offsets = {prs: i * offset_width for i, prs in enumerate(prss)}

    # Plot each PRS
    for prs in prss:
        prs_data = indirect[indirect["prs"] == prs]

        y_positions = [
            y_pos[lipid] + prs_offsets[prs] for lipid in prs_data["mediator"]
        ]

        # Plot point estimates
        ax.scatter(
            prs_data["coef"].values,
            y_positions,
            s=60,
            alpha=0.7,
            label=prs.replace("_PRS", ""),
            color=prs_colors[prs],
            zorder=3,
        )

        # Plot confidence intervals
        for (_, row), y in zip(prs_data.iterrows(), y_positions, strict=False):
            ax.plot(
                [row["CI[2.5%]"], row["CI[97.5%]"]],
                [y, y],
                alpha=0.5,
                linewidth=2,
                color=prs_colors[prs],
                zorder=2,
            )

    # Add reference line at 0
    ax.axvline(0, color="#2b2b2b", linestyle="--", linewidth=1, alpha=0.5, zorder=1)

    # Set y-axis to show lipid names
    ax.set_yticks([i + offset_width for i in range(len(lipids))])
    ax.set_yticklabels(lipids)

    ax.set_xlabel("Indirect Effect Coefficient (95% CI)", fontsize=12)
    ax.set_ylabel("Lipid Mediator", fontsize=12)
    ax.set_title(
        "Mediation Analysis: Indirect Effects of PRS on Severe Psychosis Subtype "
        "Probability via Lipids",
        fontsize=13,
        pad=20,
    )
    ax.legend(title="PRS", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add grid for readability
    ax.grid(axis="x", alpha=0.3, linestyle=":", color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_mediation_heatmap(mediation_df):
    """
    Heatmap showing indirect effect sizes across PRS-lipid combinations.
    """
    # Extract indirect effects
    indirect = mediation_df.loc[(slice(None), slice(None), "Indirect"), :].copy()
    indirect = indirect.reset_index()

    # Filter out Lipid_* PRSs
    indirect = indirect[~indirect["prs"].str.startswith("Lipid_")]

    # Clean up PRS names
    indirect["prs"] = indirect["prs"].str.replace("_PRS", "")

    # Pivot to wide format
    heatmap_data = indirect.pivot(index="mediator", columns="prs", values="coef")

    # Reorder columns
    column_order = ["BD", "SCZ", "MDD"]
    heatmap_data = heatmap_data[
        [col for col in column_order if col in heatmap_data.columns]
    ]

    fig, ax = plt.subplots(figsize=(6, 10))

    # Use a custom colormap that fits your theme
    # Create colormap from blue to white to magenta
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["#3B4CC0", "#FFFFFF", "#20B2AA"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    sns.heatmap(
        heatmap_data,
        center=0,
        cmap=cmap,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Indirect Effect"},
        linewidths=0.5,
        linecolor="#D9D9D9",
    )

    ax.set_title("Mediation Analysis: Indirect Effects", fontsize=13, pad=10)
    ax.set_xlabel("PRS", fontsize=12)
    ax.set_ylabel("Lipid Mediator", fontsize=12)

    plt.tight_layout()
    return fig
