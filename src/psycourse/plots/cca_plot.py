from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from psycourse.config import PLOT_COMBINED_LIP, PLOT_COMBINED_PRS


def plot_cca_main(
    results: dict,
    y_col: str = "prob_class_5",
    jitter_y: float = 0.0,
    figsize: tuple[float, float] = (13, 11),
) -> plt.Figure:
    """
    2x2 main figure:
      A (top left)     U1 vs V1 scatter
      B (top right)    Severity ~ U1
      C (bottom left)  PRS loadings
      D (bottom right) Lipid loadings

    Args:
        results:   CCA results dict.
        y_col:     Column name for severity outcome.
        jitter_y:  Optional vertical jitter for severity scatter (e.g. 0.005).
        figsize:   Figure dimensions.
    """
    scores = results["scores"].copy()
    u = scores["PRS_CCA_Component_1"].to_numpy(float)
    v = scores["Lipid_CCA_Component_1"].to_numpy(float)
    y = scores[y_col].to_numpy(float)

    if jitter_y > 0:
        rng = np.random.default_rng(0)
        y_plot = y + rng.normal(0.0, jitter_y, size=y.shape[0])
    else:
        y_plot = y

    can_corr = float(results.get("canonical_corr_comp1", np.corrcoef(u, v)[0, 1]))
    p_coup = results.get("p_coupling_perm_comp1", None)

    prs_load = results.get("prs_loadings", pd.Series(dtype=float))
    lip_load = results.get("lipid_class_loadings", pd.Series(dtype=float))
    prs_top = _topk_signed(prs_load, 13)
    lip_top = _topk_signed(lip_load, 16)

    dark_gray = "#2b2b2b"
    light_gray = "#D9D9D9"
    y_label = "Severe psychosis probability"

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.2)

    ax_uv = fig.add_subplot(gs[0, 0])
    ax_uy = fig.add_subplot(gs[0, 1])
    ax_prs = fig.add_subplot(gs[1, 0])
    ax_lip = fig.add_subplot(gs[1, 1])

    fig.patch.set_facecolor("white")

    for ax, label in zip(
        [ax_uv, ax_uy, ax_prs, ax_lip], ["A", "B", "C", "D"], strict=False
    ):
        ax.text(
            -0.08,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    # --- Panel A: U1 vs V1 ---
    ax_uv.scatter(u, v, s=25, alpha=0.6, color=PLOT_COMBINED_PRS, edgecolors="none")
    a, b = _ols_line(u, v)
    xx = np.linspace(np.nanmin(u), np.nanmax(u), 200)
    ax_uv.plot(xx, a + b * xx, color=dark_gray, linewidth=2, alpha=0.8)
    title_a = f"U1 vs V1 (r={can_corr:.3f}"
    if p_coup is not None:
        title_a += f", p={float(p_coup):.3g}"
    title_a += ")"
    ax_uv.set_title(title_a, fontsize=11, pad=10)
    ax_uv.set_xlabel("PRS CCA Component 1 (U1)", fontsize=10)
    ax_uv.set_ylabel("Lipid CCA Component 1 (V1)", fontsize=10)
    ax_uv.grid(True, alpha=0.3, linestyle=":", color=light_gray)
    ax_uv.set_axisbelow(True)
    ax_uv.spines["top"].set_visible(False)
    ax_uv.spines["right"].set_visible(False)

    # --- Panel B: Severity ~ V1 ---
    ax_uy.scatter(
        v, y_plot, s=25, alpha=0.6, color=PLOT_COMBINED_LIP, edgecolors="none"
    )
    a, b = _ols_line(v, y)
    xx = np.linspace(np.nanmin(v), np.nanmax(v), 200)
    ax_uy.plot(xx, a + b * xx, color=dark_gray, linewidth=2, alpha=0.8)
    p_uy = results.get("p_perm_prob_on_lipid_score", None)
    title_b = "Severe psychosis probability ~ V1"
    if p_uy is not None:
        title_b += f" (p={float(p_uy):.3g})"
    ax_uy.set_title(title_b, fontsize=11, pad=10)
    ax_uy.set_xlabel("Lipid CCA Component 1 (V1)", fontsize=10)
    ax_uy.set_ylabel(y_label, fontsize=10)
    ax_uy.grid(True, alpha=0.3, linestyle=":", color=light_gray)
    ax_uy.set_axisbelow(True)
    ax_uy.spines["top"].set_visible(False)
    ax_uy.spines["right"].set_visible(False)

    # --- Panel C: PRS loadings ---
    if len(prs_top) == 0:
        ax_prs.set_axis_off()
    else:
        colors = [PLOT_COMBINED_PRS] * len(prs_top)
        ax_prs.barh(
            prs_top.index[::-1],
            prs_top.to_numpy()[::-1],
            color=colors[::-1],
            alpha=0.8,
        )
        ax_prs.axvline(0, color=dark_gray, linewidth=1.5, alpha=0.8)
        ax_prs.set_title("PRS loadings", fontsize=11, pad=10)
        ax_prs.set_xlabel("Loading on U1", fontsize=10)
        ax_prs.tick_params(axis="y", labelsize=9)
        ax_prs.grid(True, alpha=0.3, linestyle=":", color=light_gray, axis="x")
        ax_prs.set_axisbelow(True)
        ax_prs.spines["top"].set_visible(False)
        ax_prs.spines["right"].set_visible(False)

    # --- Panel D: Lipid loadings ---
    if len(lip_top) == 0:
        ax_lip.set_axis_off()
    else:
        colors = [PLOT_COMBINED_LIP] * len(lip_top)
        ax_lip.barh(
            lip_top.index[::-1],
            lip_top.to_numpy()[::-1],
            color=colors[::-1],
            alpha=0.8,
        )
        ax_lip.axvline(0, color=dark_gray, linewidth=1.5, alpha=0.8)
        ax_lip.set_title("Lipid-class loadings", fontsize=11, pad=10)
        ax_lip.set_xlabel("Loading on V1", fontsize=10)
        ax_lip.tick_params(axis="y", labelsize=9)
        ax_lip.grid(True, alpha=0.3, linestyle=":", color=light_gray, axis="x")
        ax_lip.set_axisbelow(True)
        ax_lip.spines["top"].set_visible(False)
        ax_lip.spines["right"].set_visible(False)

    return fig


########################################################################################
# HELPERS
########################################################################################


def _topk_signed(series: pd.Series, k: int) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return s
    return s.reindex(s.abs().sort_values(ascending=False).index).head(k)


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)
