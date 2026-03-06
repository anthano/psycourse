import matplotlib.pyplot as plt


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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, valid_scores_mean, "o-", color="g", label="Validation score")
    ax.set_xlabel("Number of Training Examples")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
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

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["blue", "red", "green", "orange", "purple"]
    for i in range(n_classes):
        ax.plot(
            fpr[i],
            tpr[i],
            color=colors[i % len(colors)],
            lw=2,
            label="Class {0} (AUC = {1:0.2f})".format(classes[i], roc_auc[i]),
        )
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Each Class")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig
