from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Shared palette: one colour per class (tab10 up to 10 classes) ─────────────
_TAB10 = plt.cm.tab10.colors


def _class_colors(n: int) -> list:
    return [_TAB10[i % len(_TAB10)] for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ══════════════════════════════════════════════════════════════════════════════


def plot_learning_curve(learning_curve_data: dict) -> plt.Figure:
    """
    Plot the SVM learning curve from pre-computed data.

    Args:
        learning_curve_data: dict with keys train_sizes, train_scores_mean,
        train_scores_std, valid_scores_mean, valid_scores_std.

    Returns:
        matplotlib Figure.
    """
    train_sizes = learning_curve_data["train_sizes"]
    train_scores_mean = learning_curve_data["train_scores_mean"]
    train_scores_std = learning_curve_data["train_scores_std"]
    valid_scores_mean = learning_curve_data["valid_scores_mean"]
    valid_scores_std = learning_curve_data["valid_scores_std"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.15,
        color="tomato",
    )
    ax.fill_between(
        train_sizes,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.15,
        color="steelblue",
    )
    ax.plot(
        train_sizes, train_scores_mean, "o-", color="tomato", label="Training score"
    )
    ax.plot(
        train_sizes,
        valid_scores_mean,
        "o-",
        color="steelblue",
        label="Validation score",
    )
    ax.set_xlabel("Number of Training Examples")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Learning Curve")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def plot_roc_curves(roc_data: dict) -> plt.Figure:
    """
    Plot per-class ROC curves from pre-computed data.

    Args:
        roc_data: dict with keys fpr, tpr, roc_auc, classes, n_classes.

    Returns:
        matplotlib Figure.
    """
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    roc_auc = roc_data["roc_auc"]
    classes = roc_data["classes"]
    n_classes = roc_data["n_classes"]

    colors = _class_colors(n_classes)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n_classes):
        ax.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=2,
            label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves per Class")
    ax.legend(loc="lower right", fontsize=8)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm_data: dict) -> plt.Figure:
    """
    Plot a row-normalized confusion matrix heatmap.

    Args:
        cm_data: dict with keys cm, cm_norm, classes.

    Returns:
        matplotlib Figure.
    """
    cm_norm = cm_data["cm_norm"]
    cm_raw = cm_data["cm"]
    classes = cm_data["classes"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=np.array(
            [
                [
                    f"{cm_norm[r, c]:.2f}\n(n={cm_raw[r, c]})"
                    for c in range(len(classes))
                ]
                for r in range(len(classes))
            ]
        ),
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=[f"Class {c}" for c in classes],
        yticklabels=[f"Class {c}" for c in classes],
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_per_class_metrics(report_dict: dict) -> plt.Figure:
    """
    Grouped bar chart of precision, recall, and F1-score per class.

    Args:
        report_dict: dict returned by sklearn classification_report(output_dict=True).

    Returns:
        matplotlib Figure.
    """
    # Extract only integer/string class keys (skip 'accuracy', 'macro avg', etc.)
    class_keys = [
        k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")
    ]
    classes = class_keys
    precision = [report_dict[k]["precision"] for k in classes]
    recall = [report_dict[k]["recall"] for k in classes]
    f1 = [report_dict[k]["f1-score"] for k in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 5))
    ax.bar(x - width, precision, width, label="Precision", color="steelblue")
    ax.bar(x, recall, width, label="Recall", color="tomato")
    ax.bar(x + width, f1, width, label="F1-score", color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend(loc="lower right")

    # Macro averages as dashed reference lines
    macro = report_dict.get("macro avg", {})
    for val, color, ls in [
        (macro.get("precision"), "steelblue", "--"),
        (macro.get("recall"), "tomato", "--"),
        (macro.get("f1-score"), "seagreen", "--"),
    ]:
        if val is not None:
            ax.axhline(val, color=color, linestyle=ls, linewidth=0.8, alpha=0.6)

    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def plot_nested_cv_scores(nested_scores_data: dict) -> plt.Figure:
    """
    Strip plot of outer-fold balanced accuracy scores with mean ± std annotated.

    Args:
        nested_scores_data: dict with keys scores, mean, std.

    Returns:
        matplotlib Figure.
    """
    scores = nested_scores_data["scores"]
    mean = nested_scores_data["mean"]
    std = nested_scores_data["std"]

    fig, ax = plt.subplots(figsize=(4, 5))
    x = np.zeros(len(scores))
    ax.scatter(x, scores, color="steelblue", s=60, zorder=3, alpha=0.8)
    ax.errorbar(
        0,
        mean,
        yerr=std,
        fmt="D",
        color="tomato",
        capsize=6,
        capthick=1.5,
        elinewidth=1.5,
        markersize=8,
        zorder=4,
        label=f"Mean ± SD\n{mean:.3f} ± {std:.3f}",
    )
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Nested CV Scores\n(outer folds)")
    ax.legend(loc="lower center", fontsize=8)
    sns.despine(ax=ax, bottom=True)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Combined 2 × 2 figure
# ══════════════════════════════════════════════════════════════════════════════


def plot_svm_combined(
    learning_curve_data: dict,
    roc_data: dict,
    cm_data: dict,
    report_dict: dict,
) -> plt.Figure:
    """
    2 × 2 panel figure combining the four key SVM diagnostic plots:
      A (top-left)  : Learning curve
      B (top-right) : ROC curves per class
      C (bottom-left) : Normalized confusion matrix
      D (bottom-right): Per-class precision / recall / F1

    Args:
        learning_curve_data: dict from svm_model (learning curve).
        roc_data: dict from svm_model (ROC data).
        cm_data: dict from svm_model (confusion matrix data).
        report_dict: dict from svm_model (classification report).

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.38, wspace=0.35)

    # ── A: Learning curve ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    ts = learning_curve_data["train_sizes"]
    tr_m = learning_curve_data["train_scores_mean"]
    tr_s = learning_curve_data["train_scores_std"]
    va_m = learning_curve_data["valid_scores_mean"]
    va_s = learning_curve_data["valid_scores_std"]
    ax.fill_between(ts, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="tomato")
    ax.fill_between(ts, va_m - va_s, va_m + va_s, alpha=0.15, color="steelblue")
    ax.plot(ts, tr_m, "o-", color="tomato", label="Training score")
    ax.plot(ts, va_m, "o-", color="steelblue", label="Validation score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("A  Learning Curve", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # ── B: ROC curves ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    roc_auc = roc_data["roc_auc"]
    classes = roc_data["classes"]
    n_classes = roc_data["n_classes"]
    colors = _class_colors(n_classes)
    for i in range(n_classes):
        ax.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=2,
            label=f"Class {classes[i]} (AUC={roc_auc[i]:.2f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("B  ROC Curves per Class", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7)
    sns.despine(ax=ax)

    # ── C: Normalized confusion matrix ────────────────────────────────────────
    ax = axes[1, 0]
    cm_norm = cm_data["cm_norm"]
    cm_raw = cm_data["cm"]
    cls = cm_data["classes"]
    annot = np.array(
        [
            [f"{cm_norm[r, c]:.2f}\n(n={cm_raw[r, c]})" for c in range(len(cls))]
            for r in range(len(cls))
        ]
    )
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=[f"Class {c}" for c in cls],
        yticklabels=[f"Class {c}" for c in cls],
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("C  Normalized Confusion Matrix", loc="left", fontweight="bold")

    # ── D: Per-class precision / recall / F1 ──────────────────────────────────
    ax = axes[1, 1]
    class_keys = [
        k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")
    ]
    prec = [report_dict[k]["precision"] for k in class_keys]
    rec = [report_dict[k]["recall"] for k in class_keys]
    f1 = [report_dict[k]["f1-score"] for k in class_keys]
    x = np.arange(len(class_keys))
    w = 0.25
    ax.bar(x - w, prec, w, label="Precision", color="steelblue")
    ax.bar(x, rec, w, label="Recall", color="tomato")
    ax.bar(x + w, f1, w, label="F1", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cls {c}" for c in class_keys], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("D  Per-Class Metrics", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    macro = report_dict.get("macro avg", {})
    for val, color in [
        (macro.get("precision"), "steelblue"),
        (macro.get("recall"), "tomato"),
        (macro.get("f1-score"), "seagreen"),
    ]:
        if val is not None:
            ax.axhline(val, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
    sns.despine(ax=ax)

    return fig
