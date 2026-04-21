import matplotlib.pyplot as plt
import pandas as pd


def plot_stacked_histograms(df, lipid_color, prs_color, bins=20):
    """
    Plot two histograms of predicted probability of severe psychosis cluster:
    1. Total sample
    2. Lipids subsample

    Args:
        df (pd.DataFrame): DataFrame containing the full multimodal dataset.
        bins (int): Number of bins for the histograms.
        lipid_color (str): Color for the lipids subsample histogram.
        prs_color (str): Color for the total sample histogram.

    Returns:
        A figure with two stacked histograms.
    """
    # --- First dataset (PRS sample) ---
    df = df.copy()
    prs_columns = [col for col in df.columns if col.endswith("_PRS")]
    df = df.dropna(subset=prs_columns, how="all")

    fig, axes = plt.subplots(2, sharey=False)

    df["prob_class_5"].hist(bins=bins, ax=axes[0], color="#C8D4F2")
    axes[0].set_xlabel("Predicted probability of severe psychosis cluster")
    axes[0].set_ylabel("Count")
    axes[0].grid(False)
    axes[0].set_title("PRS Sample")

    # --- Second dataset (lipids subsample) ---
    data = df.copy()
    data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]
    lipid_features = [col for col in data.columns if col.startswith("gpeak")]
    relevant_cols = covariates + lipid_features + target
    data_with_lipids = data[~data[lipid_features].isna().all(axis=1)]
    analysis_data = data_with_lipids[relevant_cols].dropna().copy()

    analysis_data["prob_class_5"].hist(bins=bins, ax=axes[1], color="#4572E1")
    axes[1].set_xlabel("Predicted probability of severe psychosis cluster")
    axes[1].set_ylabel("Count")
    axes[1].grid(False)
    axes[1].set_title("Lipids Sample")

    plt.tight_layout()

    return fig
