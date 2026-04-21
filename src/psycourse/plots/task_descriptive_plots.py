import matplotlib.pyplot as plt
import pandas as pd

from psycourse.config import (
    BLD_DATA,
    BLD_RESULTS,
    PLOT_COMBINED_LIP,
    PLOT_COMBINED_PRS,
    SRC,
)
from psycourse.plots.descriptive_plots import plot_stacked_histograms

products = {
    "plots": BLD_RESULTS / "plots/descriptive/stacked_histograms.svg",
}


def task_plot_stacked_histograms(
    script_path=SRC / "plots/descriptive_plots.py",
    multimodal_data_path=BLD_DATA / "multimodal_complete_df.pkl",
    produces=products,
):
    """Generate stacked histograms for the total sample and lipids subsample."""

    df = pd.read_pickle(multimodal_data_path)

    fig = plot_stacked_histograms(
        df, lipid_color=PLOT_COMBINED_LIP, prs_color=PLOT_COMBINED_PRS, bins=20
    )

    fig.savefig(produces["plots"], bbox_inches="tight")
    plt.close(fig)
