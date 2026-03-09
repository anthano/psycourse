from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from psycourse.config import (
    PLOT_COMBINED_LIP as _COLOR_LIP,
)
from psycourse.config import (
    PLOT_COMBINED_PRS as _COLOR_PRS,
)
from psycourse.config import (
    PLOT_GRID_DARK as _GRID,
)
from psycourse.config import (
    PLOT_SPINE_COLOR as _DARK,
)


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_incremental_r2(results: dict) -> plt.Figure:
    """
    Publication-ready stacked bar chart for the residualized incremental R²
    decomposition.

    Shows three bars (residualized approach only):
      - PRS alone: single bar in PRS color
      - Lipids alone: single bar in lipid color
      - PRS + Lipids jointly: PRS contribution (bottom) stacked below lipid
        contribution (top), making near-perfect additivity immediately visible

    Significance asterisks (permutation test) are placed above each bar:
    * p<0.05, ** p<0.01, *** p<0.001.

    Args:
        results: dict returned by incremental_r2_decomposition.

    Returns:
        matplotlib Figure (~3.5 in wide, suitable for a journal single column).
    """
    # ── Values ────────────────────────────────────────────────────────────────
    dr2_prs = results["dR2_prs"]
    dr2_lip = results["dR2_lip"]
    dr2_joint = results["dR2_joint"]
    p_prs = results["p_perm_dR2_prs"]
    p_lip = results["p_perm_dR2_lip"]
    p_joint = results["p_perm_dR2_joint"]

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    # Minimum display height so the shared strip is always visible even when
    # the true shared variance is near zero. Computed after figsize is fixed.
    _min_shared_display = dr2_joint * 0.04  # 4 % of total bar height
    fig.patch.set_facecolor("white")

    bar_w = 0.50

    # ── PRS-alone bar ─────────────────────────────────────────────────────────
    ax.bar(
        0,
        dr2_prs,
        width=bar_w,
        color=_COLOR_PRS,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # ── Lipids-alone bar ──────────────────────────────────────────────────────
    ax.bar(
        1,
        dr2_lip,
        width=bar_w,
        color=_COLOR_LIP,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # ── PRS + Lipids jointly: PRS (bottom) + shared (middle) + lipid (top) ───
    ax.bar(
        2,
        dr2_prs,
        width=bar_w,
        color=_COLOR_PRS,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.bar(
        2,
        dr2_lip,
        width=bar_w,
        bottom=dr2_prs,
        color=_COLOR_LIP,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # ── Significance markers ──────────────────────────────────────────────────
    h_prs = dr2_prs
    h_lip = dr2_lip
    h_joint = dr2_joint  # total height of stacked joint bar

    y_top = max(h_prs, h_lip, h_joint)
    pad = y_top * 0.04

    for x, h, p in [(0, h_prs, p_prs), (1, h_lip, p_lip), (2, h_joint, p_joint)]:
        label = _sig_label(p)
        ax.text(
            x,
            h + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=8 if label != "ns" else 7,
            color=_DARK,
            zorder=4,
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_ylim(0, y_top * 1.22)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        ["PRS\nalone", "Lipids\nalone", "PRS + Lipids\njointly"],
        fontsize=8.5,
    )
    ax.set_ylabel(
        "ΔR² (severe psychosis\nsubtype probability)",
        fontsize=8.5,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color=_GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(_DARK)
    ax.spines["left"].set_color(_DARK)
    ax.tick_params(axis="both", labelsize=8)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(
            facecolor=_COLOR_PRS, edgecolor="none", label="PRS contribution"
        ),
        mpatches.Patch(
            facecolor=_COLOR_LIP, edgecolor="none", label="Lipid contribution"
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        loc="upper left",
        handlelength=1.4,
        handleheight=1.0,
        labelspacing=0.4,
    )

    fig.tight_layout()
    return fig
