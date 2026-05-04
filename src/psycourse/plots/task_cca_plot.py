from __future__ import annotations

import pickle

import matplotlib.pyplot as plt

from psycourse.config import BLD_RESULTS, SRC
from psycourse.plots.cca_plot import plot_cca_main

products_main = {
    "plots": BLD_RESULTS / "plots" / "cca" / "cca_main.svg",
}

products_supplement = {
    "plots": BLD_RESULTS / "plots" / "cca" / "cca_supplement.svg",
}


def task_plot_cca_main(
    script_path=SRC / "plots" / "cca_plot.py",
    results_dict_path=BLD_RESULTS / "cca_regression" / "results.pkl",
    produces=products_main,
):
    with open(results_dict_path, "rb") as file:
        results_dict = pickle.load(file)

    fig = plot_cca_main(results_dict)
    fig.savefig(products_main["plots"])
    fig.savefig(
        products_main["plots"].with_suffix(".tiff"), dpi=600, bbox_inches="tight"
    )
    plt.close(fig)
