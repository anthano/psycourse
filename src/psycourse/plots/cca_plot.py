from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cca_regression_summary(
    results: dict,
    y_col: str = "prob_class_5",
    top_k: int = 12,
    jitter_y: float = 0.0,
    figsize: tuple[float, float] = (16, 8),
) -> plt.Figure:
    """
    Expects (from your pipeline):
      results["scores"] with columns: PRS_CCA_Component_1, Lipid_CCA_Component_1, y_col
      results["canonical_corr_comp1"]
      results["p_coupling_perm_comp1"]
      results["prs_loadings"], results["lipid_class_loadings"]
      results["coupling_null"] (optional)
      results["prs_regression_null_t"] (optional)
      results["lip_regression_null_t"] (optional)
      results["reg_prob_on_prs_score"], results["reg_prob_on_lipid_score"]
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
    prs_top = _topk_signed(prs_load, top_k)
    lip_top = _topk_signed(lip_load, top_k)

    # layout: 2 rows x 4 cols
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    ax_uv, ax_u_y, ax_v_y, ax_null_coup = axes[0]
    ax_prs, ax_lip, ax_null_prs, ax_null_lip = axes[1]

    # (A) U1 vs V1
    ax_uv.scatter(u, v, s=18, alpha=0.8)
    a, b = _ols_line(u, v)
    xx = np.linspace(np.nanmin(u), np.nanmax(u), 200)
    ax_uv.plot(xx, a + b * xx)
    title = f"U1 vs V1 (r={can_corr:.3f})"
    if p_coup is not None:
        title += f", p_perm={float(p_coup):.3g}"
    ax_uv.set_title(title)
    ax_uv.set_xlabel("PRS CCA Component 1 (U1)")
    ax_uv.set_ylabel("Lipid CCA Component 1 (V1)")

    # (B) y vs U1
    ax_u_y.scatter(u, y_plot, s=18, alpha=0.8)
    a, b = _ols_line(u, y)
    xx = np.linspace(np.nanmin(u), np.nanmax(u), 200)
    ax_u_y.plot(xx, a + b * xx)
    p = results.get("p_perm_prob_on_prs_score", None)
    ax_u_y.set_title(
        f"{y_col} ~ U1" + (f" (p_perm={float(p):.3g})" if p is not None else "")
    )
    ax_u_y.set_xlabel("U1")
    ax_u_y.set_ylabel(y_col)

    # (C) y vs V1
    ax_v_y.scatter(v, y_plot, s=18, alpha=0.8)
    a, b = _ols_line(v, y)
    xx = np.linspace(np.nanmin(v), np.nanmax(v), 200)
    ax_v_y.plot(xx, a + b * xx)
    p = results.get("p_perm_prob_on_lipid_score", None)
    ax_v_y.set_title(
        f"{y_col} ~ V1" + (f" (p_perm={float(p):.3g})" if p is not None else "")
    )
    ax_v_y.set_xlabel("V1")
    ax_v_y.set_ylabel(y_col)

    # (D) Coupling null (r)
    coupling_null = results.get("coupling_null", None)
    if coupling_null is None:
        ax_null_coup.set_axis_off()
    else:
        null = np.asarray(coupling_null, float)
        null = null[np.isfinite(null)]
        ax_null_coup.hist(null, bins=40, alpha=0.85)
        ax_null_coup.axvline(can_corr, linewidth=2)
        ax_null_coup.set_title("Coupling perm null (r)")
        ax_null_coup.set_xlabel("Null canonical correlation")
        ax_null_coup.set_ylabel("Count")

    # (E) PRS loadings
    if len(prs_top) == 0:
        ax_prs.set_axis_off()
    else:
        ax_prs.barh(prs_top.index[::-1], prs_top.to_numpy()[::-1])
        ax_prs.axvline(0, linewidth=1)
        ax_prs.set_title(f"Top PRS loadings (k={len(prs_top)})")
        ax_prs.set_xlabel("Loading on U1")

    # (F) Lipid loadings
    if len(lip_top) == 0:
        ax_lip.set_axis_off()
    else:
        ax_lip.barh(lip_top.index[::-1], lip_top.to_numpy()[::-1])
        ax_lip.axvline(0, linewidth=1)
        ax_lip.set_title(f"Top lipid-class loadings (k={len(lip_top)})")
        ax_lip.set_xlabel("Loading on V1")

    # helper to extract observed t
    def _get_t(res_key: str, col_name: str) -> float:
        res = results.get(res_key, None)
        if res is None:
            return float("nan")
        try:
            tv = getattr(res, "tvalues", None)
            if tv is None:
                return float("nan")
            # tv can be array-like or Series
            if isinstance(tv, (pd.Series, dict)):
                return float(tv.get(col_name, np.nan))
            # fallback: try params index ordering
            return float(tv[1])  # intercept, slope
        except Exception:
            return float("nan")

    # (G) PRS regression null (t)
    prs_null_t = results.get("prs_regression_null_t", None)
    t_obs_prs = _get_t("reg_prob_on_prs_score", "PRS_CCA_Component_1")
    if prs_null_t is None:
        ax_null_prs.set_axis_off()
    else:
        null = np.asarray(prs_null_t, float)
        null = null[np.isfinite(null)]
        ax_null_prs.hist(null, bins=40, alpha=0.85)
        if np.isfinite(t_obs_prs):
            ax_null_prs.axvline(t_obs_prs, linewidth=2)
        ax_null_prs.set_title("PRS score perm null (t)")
        ax_null_prs.set_xlabel("Null t-statistic")
        ax_null_prs.set_ylabel("Count")

    # (H) Lipid regression null (t)
    lip_null_t = results.get("lip_regression_null_t", None)
    t_obs_lip = _get_t("reg_prob_on_lipid_score", "Lipid_CCA_Component_1")
    if lip_null_t is None:
        ax_null_lip.set_axis_off()
    else:
        null = np.asarray(lip_null_t, float)
        null = null[np.isfinite(null)]
        ax_null_lip.hist(null, bins=40, alpha=0.85)
        if np.isfinite(t_obs_lip):
            ax_null_lip.axvline(t_obs_lip, linewidth=2)
        ax_null_lip.set_title("Lipid score perm null (t)")
        ax_null_lip.set_xlabel("Null t-statistic")
        ax_null_lip.set_ylabel("Count")

    return fig


########################################################################################
# HELPER
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
