import matplotlib.pyplot as plt
import pandas as pd


def plot_stacked_histograms(df, bins=20):
    """
    Plot two histograms of predicted probability of severe psychosis cluster:
    1. Total sample
    2. Lipids subsample

    Args:
        df (pd.DataFrame): DataFrame containing the full multimodal dataset.
        bins (int): Number of bins for the histograms.

    Returns:
        A figure with two stacked histograms.
    """
    # --- First dataset (total sample) ---
    fig, axes = plt.subplots(2, sharey=False)

    df["prob_class_5"].hist(bins=bins, ax=axes[0], color="#F39C12")
    axes[0].set_xlabel("Predicted probability of severe psychosis cluster")
    axes[0].set_ylabel("Count")
    axes[0].grid(False)
    axes[0].set_title("Total sample")

    # --- Second dataset (lipids subsample) ---
    data = df.copy()
    data["sex"] = data["sex"].map({"F": 0, "M": 1}).astype(pd.Int8Dtype())
    covariates = ["age", "bmi", "sex"]
    target = ["prob_class_5"]
    lipid_features = [col for col in data.columns if col.startswith("gpeak")]
    relevant_cols = covariates + lipid_features + target
    data_with_lipids = data[~data[lipid_features].isna().all(axis=1)]
    analysis_data = data_with_lipids[relevant_cols].dropna().copy()

    analysis_data["prob_class_5"].hist(bins=bins, ax=axes[1], color="#7E1E9C")
    axes[1].set_xlabel("Predicted probability of severe psychosis cluster")
    axes[1].set_ylabel("Count")
    axes[1].grid(False)
    axes[1].set_title("Lipids subsample")

    plt.tight_layout()

    return fig
