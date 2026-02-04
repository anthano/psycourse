from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from psycourse.config import BLD_RESULTS, SRC, WRITING
from psycourse.plots.mediation_plots import (
    plot_mediation_heatmap,
    plot_mediation_results,
)

products_results_plot = {
    "plots": BLD_RESULTS / "plots" / "mediation" / "mediation_results.svg",
    "plots_for_writing": WRITING / "plots" / "mediation" / "mediation_results.svg",
}


def task_plot_mediation_results(
    script_path=SRC / "plots" / "mediation_plots.py",
    results_df_path=BLD_RESULTS
    / "mediation_analysis"
    / "mediation_analysis_results.pkl",
    produces=products_results_plot,
):
    results_df = pd.read_pickle(results_df_path)

    fig = plot_mediation_results(results_df)
    fig.savefig(products_results_plot["plots"])
    plt.close(fig)

    fig.savefig(products_results_plot["plots_for_writing"])
    plt.close(fig)


########################################################################################

products_heatmap = {
    "plots": BLD_RESULTS / "plots" / "mediation" / "mediation_heatmap.svg",
    "plots_for_writing": WRITING / "plots" / "mediation" / "mediation_heatmap.svg",
}


def task_plot_mediation_heatmap(
    script_path=SRC / "plots" / "mediation_plots.py",
    results_df_path=BLD_RESULTS
    / "mediation_analysis"
    / "mediation_analysis_results.pkl",
    produces=products_heatmap,
):
    results_df = pd.read_pickle(results_df_path)

    fig = plot_mediation_heatmap(results_df)
    fig.savefig(products_heatmap["plots"])
    plt.close(fig)

    fig.savefig(products_heatmap["plots_for_writing"])
    plt.close(fig)
