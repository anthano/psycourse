from __future__ import annotations

import pickle

import matplotlib.pyplot as plt

from psycourse.config import BLD_RESULTS, SRC
from psycourse.plots.incremental_r2_plot import plot_incremental_r2

_PLOT_DIR = BLD_RESULTS / "plots" / "incremental_r2"

products = {
    "plot": _PLOT_DIR / "incremental_r2_decomposition.svg",
}


def task_plot_incremental_r2(
    script_path=SRC / "plots" / "incremental_r2_plot.py",
    results_path=BLD_RESULTS / "incremental_r2" / "incremental_r2_results.pkl",
    produces=products,
) -> None:
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    fig = plot_incremental_r2(results)
    fig.savefig(products["plot"])
    fig.savefig(products["plot"].with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
