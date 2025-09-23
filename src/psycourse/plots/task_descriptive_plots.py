import matplotlib.pyplot as plt
import pandas as pd

from psycourse.config import BLD_DATA, BLD_RESULTS, SRC
from psycourse.plots.descriptive_plots import plot_stacked_histograms


def task_plot_stacked_histograms(
    script_path=SRC / "plots/descriptive_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=BLD_RESULTS / "plots/descriptive/stacked_histograms.svg",
):
    """Generate stacked histograms for the total sample and lipids subsample."""

    # Load the multimodal dataset
    df = pd.read_pickle(multimodal_data_path)

    # Generate the stacked histograms
    fig = plot_stacked_histograms(df, bins=20)

    # Save the figure
    fig.savefig(produces, bbox_inches="tight")
    plt.close(fig)
