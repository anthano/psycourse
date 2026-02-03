from __future__ import annotations

import pickle

import matplotlib.pyplot as plt

from psycourse.config import BLD_RESULTS, SRC, WRITING
from psycourse.plots.cca_plot import plot_cca_regression_summary

products = {
    "plots": BLD_RESULTS / "plots" / "cca" / "cca_regression.svg",
    "plots_for_writing": WRITING / "plots" / "cca" / "cca_regression.svg",
}


def task_plot_cca(
    script_path=SRC / "plots" / "cca_plot.py",
    results_dict_path=BLD_RESULTS / "cca_regression" / "results.pkl",
    produces=products,
):
    with open(results_dict_path, "rb") as file:
        results_dict = pickle.load(file)

    fig = plot_cca_regression_summary(results_dict)
    fig.savefig(products["plots"])
    plt.close(fig)

    fig.savefig(products["plots_for_writing"])
    plt.close(fig)
