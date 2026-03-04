from __future__ import annotations

import pickle

import matplotlib.pyplot as plt

from psycourse.config import BLD_RESULTS, SRC, WRITING
from psycourse.plots.incremental_r2_plot import plot_incremental_r2

_PLOT_DIR = BLD_RESULTS / "plots" / "incremental_r2"
_WRITING_DIR = WRITING / "plots" / "incremental_r2"

products = {
    "plot": _PLOT_DIR / "incremental_r2_decomposition.svg",
    "plot_for_writing": _WRITING_DIR / "incremental_r2_decomposition.png",
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
    plt.close(fig)

    fig = plot_incremental_r2(results)
    fig.savefig(products["plot_for_writing"], dpi=300)
    plt.close(fig)
