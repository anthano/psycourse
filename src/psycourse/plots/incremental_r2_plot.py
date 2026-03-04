from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette
_COLOR_PRS = "#0072B2"  # blue
_COLOR_LIP = "#E69F00"  # amber/orange
_COLOR_SHARED = "#999999"  # neutral grey
_HATCH = "///"


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _draw_stacked_bar(
    ax: plt.Axes,
    x: float,
    segments: list[tuple[float, str]],
    bar_width: float,
    use_hatch: bool = False,
) -> float:
    """
    Draw a stacked bar composed of (height, color) segments from bottom to top.
    Segments with non-positive height are skipped.
    Returns the total bar height (for significance marker placement).
    """
    bottom = 0.0
    for height, color in segments:
        if height <= 0.0:
            continue
        if use_hatch:
            ax.bar(
                x=x,
                height=height,
                bottom=bottom,
                width=bar_width,
                facecolor=color,
                edgecolor=color,
                hatch=_HATCH,
                alpha=0.6,
                linewidth=0,
                zorder=3,
            )
        else:
            ax.bar(
                x=x,
                height=height,
                bottom=bottom,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        bottom += height
    return bottom


def plot_incremental_r2(results: dict) -> plt.Figure:
    """
    Publication-ready stacked bar chart for incremental R² decomposition.

    Shows three comparisons on the x-axis — PRS alone, Lipids alone, and
    PRS + Lipids jointly — each as a pair of adjacent bars: the block-specific
    residualized approach (solid fill) and the traditional nested approach
    (hatched, semi-transparent).

    Each bar is stacked with up to three segments:
      - PRS contribution (blue)
      - Lipid contribution (amber)
      - Shared variance (grey; joint bar only, = dR2_joint − dR2_prs − dR2_lip)

    Significance asterisks (permutation test) are placed above each bar.

    Args:
        results: dict returned by incremental_r2_decomposition.

    Returns:
        matplotlib Figure (~3.5 in wide, suitable for a journal single column).
    """
    # ── Residualized values ───────────────────────────────────────────────────
    dr2_prs = results["dR2_prs"]
    dr2_lip = results["dR2_lip"]
    dr2_joint = results["dR2_joint"]
    shared = max(0.0, dr2_joint - dr2_prs - dr2_lip)

    p_prs = results["p_perm_dR2_prs"]
    p_lip = results["p_perm_dR2_lip"]
    p_joint = results["p_perm_dR2_joint"]

    # ── Traditional values ────────────────────────────────────────────────────
    dr2_prs_t = results["dR2_prs_trad"]
    dr2_lip_t = results["dR2_lip_trad"]
    dr2_joint_t = results["dR2_joint_trad"]
    shared_t = max(0.0, dr2_joint_t - dr2_prs_t - dr2_lip_t)

    p_prs_t = results["p_perm_dR2_prs_trad"]
    p_lip_t = results["p_perm_dR2_lip_trad"]
    p_joint_t = results["p_perm_dR2_joint_trad"]

    # ── Bar positions ─────────────────────────────────────────────────────────
    bar_w = 0.28
    off = 0.16  # offset from group centre to each bar's centre

    centers = np.array([0.0, 1.0, 2.0])
    pos_r = centers - off  # residualized (left bar of each pair)
    pos_t = centers + off  # traditional  (right bar of each pair)

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 3.8))
    fig.patch.set_facecolor("white")

    # ── PRS-alone group ───────────────────────────────────────────────────────
    h_prs_r = _draw_stacked_bar(ax, pos_r[0], [(dr2_prs, _COLOR_PRS)], bar_w)
    h_prs_t = _draw_stacked_bar(
        ax, pos_t[0], [(dr2_prs_t, _COLOR_PRS)], bar_w, use_hatch=True
    )

    # ── Lipids-alone group ────────────────────────────────────────────────────
    h_lip_r = _draw_stacked_bar(ax, pos_r[1], [(dr2_lip, _COLOR_LIP)], bar_w)
    h_lip_t = _draw_stacked_bar(
        ax, pos_t[1], [(dr2_lip_t, _COLOR_LIP)], bar_w, use_hatch=True
    )

    # ── PRS + Lipids jointly group ────────────────────────────────────────────
    h_jnt_r = _draw_stacked_bar(
        ax,
        pos_r[2],
        [
            (dr2_prs, _COLOR_PRS),
            (dr2_lip, _COLOR_LIP),
            (shared, _COLOR_SHARED),
        ],
        bar_w,
    )
    h_jnt_t = _draw_stacked_bar(
        ax,
        pos_t[2],
        [
            (dr2_prs_t, _COLOR_PRS),
            (dr2_lip_t, _COLOR_LIP),
            (shared_t, _COLOR_SHARED),
        ],
        bar_w,
        use_hatch=True,
    )

    # ── Significance markers ──────────────────────────────────────────────────
    y_top = max(h_prs_r, h_prs_t, h_lip_r, h_lip_t, h_jnt_r, h_jnt_t)
    pad = y_top * 0.04

    for x, h, p in [
        (pos_r[0], h_prs_r, p_prs),
        (pos_t[0], h_prs_t, p_prs_t),
        (pos_r[1], h_lip_r, p_lip),
        (pos_t[1], h_lip_t, p_lip_t),
        (pos_r[2], h_jnt_r, p_joint),
        (pos_t[2], h_jnt_t, p_joint_t),
    ]:
        label = _sig_label(p)
        ax.text(
            x,
            h + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=8 if label != "ns" else 7,
            color="#2b2b2b",
            zorder=4,
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_ylim(0, y_top * 1.25)
    ax.set_xlim(-0.52, 2.52)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        ["PRS\nalone", "Lipids\nalone", "PRS + Lipids\njointly"],
        fontsize=8.5,
    )
    ax.set_ylabel(
        "ΔR² (severe psychosis\nsubtype probability)",
        fontsize=8.5,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2b2b2b")
    ax.spines["left"].set_color("#2b2b2b")
    ax.tick_params(axis="both", labelsize=8)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(
            facecolor=_COLOR_PRS, edgecolor="none", label="PRS contribution"
        ),
        mpatches.Patch(
            facecolor=_COLOR_LIP, edgecolor="none", label="Lipid contribution"
        ),
        mpatches.Patch(
            facecolor=_COLOR_SHARED, edgecolor="none", label="Shared variance"
        ),
        mpatches.Patch(
            facecolor="#cccccc",
            edgecolor="#888888",
            hatch=_HATCH,
            alpha=0.6,
            label="Traditional (raw)",
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
