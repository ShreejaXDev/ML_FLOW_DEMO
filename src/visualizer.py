"""
Visualization Module for Model Evaluation and Comparison.

Generates publication-quality charts using Matplotlib and Seaborn:
1. Dataset class distribution chart
2. Accuracy comparison chart
3. Precision comparison chart
4. Recall comparison chart
5. F1-score comparison chart
6. ROC curves comparison chart
7. Precision-Recall curves comparison chart
8. Confusion matrices for all models
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from config.config import RESULTS_DIR

# Set global aesthetic style
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_class_distribution(
    normal_count: int,
    fraud_count: int,
    output_path: Path = RESULTS_DIR / "class_distribution.png",
) -> Path:
    """
    Generate a dual-panel chart illustrating extreme class imbalance.

    Parameters
    ----------
    normal_count : int
        Count of normal transactions.
    fraud_count : int
        Count of fraudulent transactions.
    output_path : Path
        Destination path.

    Returns
    -------
    Path
        Saved image path.
    """
    total = normal_count + fraud_count
    normal_pct = (normal_count / total) * 100
    fraud_pct = (fraud_count / total) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Bar chart with log scale
    labels = ["Normal (0)", "Fraud (1)"]
    counts = [normal_count, fraud_count]
    colors = ["#2b5c8f", "#d9534f"]

    bars = axes[0].bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_title("Transaction Counts (Logarithmic Scale)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Transactions (Log Scale)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    for bar, count, pct in zip(bars, counts, [normal_pct, fraud_pct]):
        y_pos = bar.get_height()
        axes[0].annotate(
            f"{count:,}\n({pct:.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel 2: Percentage breakdown donut chart
    wedge_props = {"edgecolor": "white", "linewidth": 2, "width": 0.4}
    axes[1].pie(
        [normal_count, fraud_count],
        labels=[f"Normal: {normal_pct:.2f}%", f"Fraud: {fraud_pct:.2f}%"],
        colors=colors,
        startangle=140,
        wedgeprops=wedge_props,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    axes[1].set_title("Extreme Class Imbalance (Prevalence)", fontsize=13, fontweight="bold", pad=12)

    plt.suptitle(
        "Credit Card Fraud Dataset: Why 99.8% Accuracy is Trivial",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_single_metric_comparison(
    models_metrics: Dict[str, Dict[str, float]],
    metric_key: str,
    metric_label: str,
    output_path: Path,
    higher_is_better: bool = True,
    highlight_dummy: bool = True,
) -> Path:
    """
    Plot a bar chart comparing a single metric across all models.
    """
    model_names = list(models_metrics.keys())
    values = [models_metrics[m][metric_key] for m in model_names]

    # Color scheme: grey/orange for Dummy, blue for LR, green for RF
    palette = []
    for m in model_names:
        if "Dummy" in m:
            palette.append("#e67e22" if metric_key == "accuracy" else "#95a5a6")
        elif "Logistic" in m:
            palette.append("#2980b9")
        else:
            palette.append("#27ae60")

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(model_names, values, color=palette, width=0.45, edgecolor="black", linewidth=1.1)

    plt.title(f"Model Comparison: {metric_label}", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel(metric_label, fontsize=12)
    plt.ylim(0, max(1.15, max(values) * 1.15))
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars, values):
        plt.annotate(
            f"{val:.4f} ({val*100:.1f}%)" if val <= 1.0 else f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Educational annotation for Accuracy Paradox
    if metric_key == "accuracy" and highlight_dummy:
        plt.figtext(
            0.5, -0.05,
            "[!] Note: Dummy Classifier reaches ~99.8% accuracy simply by guessing 'Normal' for everything!",
            ha="center", fontsize=10, fontstyle="italic", color="#c0392b", weight="bold"
        )
    elif metric_key == "recall":
        plt.figtext(
            0.5, -0.05,
            "[*] Notice: Dummy Classifier has 0.0% Recall — it fails to identify a single fraudulent transaction!",
            ha="center", fontsize=10, fontstyle="italic", color="#27ae60", weight="bold"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_roc_curves(
    curves_data: List[Dict[str, Any]],
    output_path: Path = RESULTS_DIR / "roc_curves_comparison.png",
) -> Path:
    """
    Plot ROC Curves for all models on a single figure.
    """
    plt.figure(figsize=(8.5, 6.5))

    # Diagonal baseline (random guess)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Baseline (AUC = 0.5000)", alpha=0.7)

    colors = ["#e67e22", "#2980b9", "#27ae60"]
    for i, data in enumerate(curves_data):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc = data["auc"]
        name = data["name"]
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, lw=2.2)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Recall / TPR)", fontsize=11, labelpad=8)
    plt.title("ROC Curves Comparison: All Models", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_pr_curves(
    curves_data: List[Dict[str, Any]],
    prevalence: float,
    output_path: Path = RESULTS_DIR / "pr_curves_comparison.png",
) -> Path:
    """
    Plot Precision-Recall Curves for all models on a single figure.
    """
    plt.figure(figsize=(8.5, 6.5))

    # Baseline: no-skill classifier precision = positive class prevalence
    plt.plot(
        [0, 1],
        [prevalence, prevalence],
        "k--",
        lw=1.5,
        label=f"No-Skill Baseline (AP = {prevalence:.4f})",
        alpha=0.7,
    )

    colors = ["#e67e22", "#2980b9", "#27ae60"]
    for i, data in enumerate(curves_data):
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        ap = data["ap"]
        name = data["name"]
        color = colors[i % len(colors)]
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {ap:.4f})", color=color, lw=2.2)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Fraud Detected)", fontsize=11, labelpad=8)
    plt.ylabel("Precision (Fraud Identification Accuracy)", fontsize=11, labelpad=8)
    plt.title("Precision-Recall Curves Comparison (Gold Standard for Imbalance)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Path,
) -> Path:
    """
    Plot a styled Confusion Matrix heatmap with counts and cell percentages.
    """
    tn, fp, fn, tp = cm.ravel()
    total = np.sum(cm)

    cm_labels = [
        [f"TN\n{tn:,}\n({(tn/total)*100:.2f}%)", f"FP\n{fp:,}\n({(fp/total)*100:.2f}%)"],
        [f"FN\n{fn:,}\n({(fn/total)*100:.2f}%)", f"TP\n{tp:,}\n({(tp/total)*100:.2f}%)"],
    ]

    plt.figure(figsize=(6.5, 5.2))
    cmap = "Blues" if "Logistic" in model_name else ("Greens" if "Forest" in model_name else "Oranges")
    sns.heatmap(
        cm,
        annot=np.array(cm_labels),
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        xticklabels=["Predicted Normal (0)", "Predicted Fraud (1)"],
        yticklabels=["Actual Normal (0)", "Actual Fraud (1)"],
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("Actual Class", fontsize=11, labelpad=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path

