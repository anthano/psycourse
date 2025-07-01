import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

############# Lipids ###################


def plot_univariate_lipid_regression(top20):
    """Plot the top 20 lipids associated with cluster 5 probability using regression
    coefficients and FDR values.
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
        "Top 20 Lipid Associations with Cluster 5 Probability "
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


def plot_corr_matrix_lipid_top20(multimodal_df, top20_results_df):
    """Plot a correlation matrix for the top 20 lipids.

    Args:
        multimodal_df (pd.DataFrame): DataFrame containing lipidomic data (and more).
        top20_results_df (pd.DataFrame): DataFrame containing top 20 results with
        lipids in index.

    Returns:
        None: Displays a correlation matrix plot.
    """
    top20_df = multimodal_df[multimodal_df.columns.intersection(top20_results_df.index)]
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


#################### PRS ###################


def plot_univariate_prs_regression(prs_results):
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
    plt.title("PRS Associations with Cluster 5 Probability (covariates: age, sex, bmi)")
    cbar = plt.colorbar(sc)
    cbar.set_label("-log10(FDR)")
    plt.tight_layout()
    # plt.show()


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
