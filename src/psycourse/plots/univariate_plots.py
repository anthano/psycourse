import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

############# Lipids ###################


def plot_univariate_lipid_regression(lipid_results_top20, cleaned_annotation_df):
    """
    Plot lipid GLM coefficients with 95% CIs.
    Points colored by -log10(FDR); colorbar shows FDR (q) ticks.
    """
    mapping = cleaned_annotation_df["lipid_species"].to_dict()
    lipid_results_top20 = lipid_results_top20.rename(index=mapping)
    lipid_sorted = lipid_results_top20.sort_values("FDR").copy()
    y = np.arange(len(lipid_sorted))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # error bars (95% CI)
    xerr = np.vstack(
        [
            lipid_sorted["coef"] - lipid_sorted["ci_low"],
            lipid_sorted["ci_high"] - lipid_sorted["coef"],
        ]
    )
    ax.errorbar(
        lipid_sorted["coef"],
        y,
        xerr=xerr,
        fmt="o",
        ms=6,
        mec="k",
        mfc="white",
        ecolor="k",
        elinewidth=1,
        capsize=3,
        capthick=1,
        zorder=2,
    )

    # scatter colored by -log10(FDR), but show FDR ticks on the colorbar
    neglogq = -np.log10(lipid_sorted["FDR"].clip(lower=np.finfo(float).tiny))
    vmin = 1.0
    vmax = float(np.nanmax(neglogq))
    sc = ax.scatter(
        lipid_sorted["coef"],
        y,
        c=neglogq,
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        s=90,
        edgecolor="k",
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
    ax.set_ylabel("Lipids")
    ax.set_title("Lipid Associations with Severe Psychosis Cluster Probability")

    # Colorbar showing FDR ticks
    cbar = plt.colorbar(sc)
    cbar.set_label("FDR (q)")
    ticks_all = np.array([1, 1.30103, 2, 3])
    ticks = ticks_all[ticks_all <= vmax]
    cbar.set_ticks(ticks)  # q= 0.1, 0.05, 0.01
    cbar.ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(
            lambda t, _: f"{10**(-t):.2g}" if 10 ** (-t) >= 0.01 else f"{10**(-t):.1e }"
        )
    )
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

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Draw lines
    for i, row in lipid_class_results_sorted.iterrows():
        plt.plot([0, row["coef"]], [i, i], "k-", lw=0.5)

    # Draw points
    sc = plt.scatter(
        lipid_class_results_sorted["coef"],
        lipid_class_results_sorted.index,
        c=-np.log10(lipid_class_results_sorted["FDR"]),
        cmap="plasma",
        s=80,
        edgecolor="k",
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Regression Coefficient")
    plt.ylabel("Lipid Class")
    plt.title(
        "Lipid Class Association with Cluster 5 Probability "
        "(covariates: age, sex, bmi)"
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("-log10(FDR)")
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

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Draw lines
    for i, row in top20_sorted.iterrows():
        plt.plot([0, row["coef"]], [i, i], "k-", lw=0.5)

    # Draw points
    sc = plt.scatter(
        top20_sorted["coef"],
        top20_sorted.index,
        c=-np.log10(top20_sorted["FDR"]),
        cmap="plasma",
        s=80,
        edgecolor="k",
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Regression Coefficient")
    plt.ylabel("Lipid")
    plt.title(
        "Top 20 Lipid Associations with Cluster 5 Probability Top50 vs. Bottom 50 "
        "(covariates: age, sex, bmi)"
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("-log10(FDR)")
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
      color = -log10(FDR)  (colorbar shows FDR ticks)
      size = n_in_class
    """
    if enrich_df.empty:
        raise ValueError("enrich_df is empty.")

    df = enrich_df.sort_values("FDR").head(top_n).iloc[::-1].copy()  # most sig at top

    # Slightly truncate plasma to avoid very bright top end
    plasma_trunc = LinearSegmentedColormap.from_list(
        "plasma_trunc", plt.cm.plasma(np.linspace(0.0, 0.92, 256))
    )

    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        },
    )
    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter
    sc = ax.scatter(
        df["ES"],
        df.index,
        s=20 + 6 * df["n_in_class"],  # size by class size
        c=df["-log10(FDR)"],
        cmap=plasma_trunc,
        edgecolor="k",
        linewidth=0.6,
        zorder=3,
    )

    # Zero line = no enrichment
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    # Labels / grid
    ax.set_xlabel("Enrichment score (Δ mean rank; >0 = enriched)")
    ax.set_ylabel("Lipid class")
    ax.set_title(title)
    ax.grid(True, axis="x", color="0.9")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    sns.despine(left=True)

    # Nice symmetric x-lims with padding
    span = np.nanmax(np.abs(df["ES"]))
    pad = 0.15 * max(0.01, span)
    ax.set_xlim(-span - pad, span + pad)

    # Colorbar formatted in FDR (q)
    cbar = plt.colorbar(sc)
    cbar.set_label("FDR (q)")

    def _fmt(t, _pos):
        q = 10 ** (-t)
        return f"{q:.2g}" if q >= 0.01 else f"{q:.1e}"

    vmax = float(df["-log10(FDR)"].max())
    ticks_all = np.array([0, 1, 1.30103, 2, 3])  # q=1, 0.1, 0.05, 0.01, 0.001
    ticks = ticks_all[ticks_all <= vmax]
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt))

    plt.tight_layout()
    return fig, ax


########################################################################################
# PRS
########################################################################################


def plot_univariate_prs_regression(prs_results):
    """
    Plot PRS GLM coefficients with 95% CIs.
    Points colored by -log10(FDR); colorbar shows FDR (q) ticks.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    prs_sorted = prs_results.sort_values("FDR").copy()
    y = np.arange(len(prs_sorted))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # error bars (95% CI)
    xerr = np.vstack(
        [
            prs_sorted["coef"] - prs_sorted["ci_low"],
            prs_sorted["ci_high"] - prs_sorted["coef"],
        ]
    )
    ax.errorbar(
        prs_sorted["coef"],
        y,
        xerr=xerr,
        fmt="o",
        ms=6,
        mec="k",
        mfc="white",
        ecolor="k",
        elinewidth=1,
        capsize=3,
        capthick=1,
        zorder=2,
    )

    # scatter colored by -log10(FDR), but show FDR ticks on the colorbar
    neglogq = -np.log10(prs_sorted["FDR"].clip(lower=np.finfo(float).tiny))
    vmin = 1.0
    vmax = float(np.nanmax(neglogq))
    sc = ax.scatter(
        prs_sorted["coef"],
        y,
        c=neglogq,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
        s=90,
        edgecolor="k",
        zorder=3,
    )

    # zero (null) line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    # y labels, most significant at the TOP
    ax.set_yticks(y, prs_sorted.index)
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
    ax.set_ylabel("PRS")
    ax.set_title("PRS Associations with Severe Psychosis Cluster Probability")

    # colorbar showing FDR
    cbar = plt.colorbar(sc)
    cbar.set_label("FDR (q)")
    ticks_all = np.array([1, 1.30103, 2])
    ticks = ticks_all[ticks_all <= vmax]
    cbar.set_ticks(ticks)  # q= 0.1, 0.05, 0.01
    cbar.ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(
            lambda t, _: f"{10**(-t):.2g}" if 10 ** (-t) >= 0.01 else f"{10**(-t):.1e }"
        )
    )

    plt.tight_layout()

    return fig, ax


def plot_univariate_prs_extremes(prs_results):
    # Sort PRS by FDR to fix order
    prs_sorted = prs_results.sort_values("FDR")

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Draw lines
    for i, row in prs_sorted.iterrows():
        plt.plot([0, row["coef"]], [i, i], "k-", lw=0.5)

    # Draw points
    sc = plt.scatter(
        prs_sorted["coef"],
        prs_sorted.index,
        c=-np.log10(prs_sorted["FDR"]),
        cmap="plasma",
        s=80,
        edgecolor="k",
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Regression Coefficient")
    plt.ylabel("PRS")
    plt.title(
        "PRS Association with Cluster5Prob for Top50 vs. Bottom 50 (cov: age, sex, bmi)"
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("-log10(FDR)")
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

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

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
    ax.set_ylabel("PRS")
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
