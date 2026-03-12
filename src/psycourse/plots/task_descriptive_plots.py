import matplotlib.pyplot as plt
import pandas as pd

from psycourse.config import (
    BLD_DATA,
    BLD_RESULTS,
    PLOT_COMBINED_LIP,
    PLOT_COMBINED_PRS,
    SRC,
    WRITING,
)
from psycourse.plots.descriptive_plots import plot_stacked_histograms

products = {
    "plots": BLD_RESULTS / "plots/descriptive/stacked_histograms.svg",
    "plots_for_writing": WRITING / "plots/descriptive/stacked_histograms.svg",
}


def task_plot_stacked_histograms(
    script_path=SRC / "plots/descriptive_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=products,
):
    """Generate stacked histograms for the total sample and lipids subsample."""

    # Load the multimodal dataset
    df = pd.read_pickle(multimodal_data_path)

    # Generate the stacked histograms
    fig = plot_stacked_histograms(
        df, lipid_color=PLOT_COMBINED_LIP, prs_color=PLOT_COMBINED_PRS, bins=20
    )

    # Save the figure
    fig.savefig(produces["plots"], bbox_inches="tight")
    fig.savefig(produces["plots_for_writing"], bbox_inches="tight")
    plt.close(fig)
