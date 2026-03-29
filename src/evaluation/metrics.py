"""
src/evaluation/metrics.py
--------------------------
Evaluation utilities: log-loss, confusion matrix, per-class diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    classification_report,
)

from src.data.dataset import CLASS_NAMES, NUM_CLASSES


def compute_log_loss(
    true_labels: pd.DataFrame | np.ndarray,
    pred_probs: np.ndarray,
) -> float:
    """Compute multi-class log-loss (competition metric).

    Args:
        true_labels: One-hot encoded labels, shape (n, 8) or class indices (n,).
        pred_probs: Predicted probabilities, shape (n, 8).

    Returns:
        Scalar log-loss value.
    """
    if isinstance(true_labels, pd.DataFrame):
        true_labels = true_labels[CLASS_NAMES].values

    if true_labels.ndim == 2:
        true_indices = true_labels.argmax(axis=1)
    else:
        true_indices = true_labels

    return log_loss(true_indices, pred_probs, labels=list(range(NUM_CLASSES)))


def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    normalize: bool = True,
    output_path: str | None = None,
) -> None:
    """Plot confusion matrix from predicted probabilities."""
    true_cls = true_labels.argmax(axis=1) if true_labels.ndim == 2 else true_labels
    pred_cls = pred_probs.argmax(axis=1)

    cm = confusion_matrix(true_cls, pred_cls, labels=list(range(NUM_CLASSES)))
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved confusion matrix → {output_path}")
    plt.show()


def plot_per_class_log_loss(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_path: str | None = None,
) -> pd.Series:
    """Compute and plot per-class log-loss contribution."""
    true_cls = true_labels.argmax(axis=1) if true_labels.ndim == 2 else true_labels

    per_class_ll = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = true_cls == i
        if mask.sum() == 0:
            continue
        ll = log_loss(
            true_cls[mask],
            pred_probs[mask],
            labels=list(range(NUM_CLASSES)),
        )
        per_class_ll[cls] = ll

    ll_series = pd.Series(per_class_ll).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ll_series.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Per-class log-loss")
    ax.set_ylabel("Log-loss")
    ax.set_xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print("\nPer-class log-loss:")
    print(ll_series.to_string())
    return ll_series


def full_diagnostics(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_dir: str = "data/processed",
) -> dict[str, float]:
    """Run full diagnostic suite and save plots."""
    ll = compute_log_loss(true_labels, pred_probs)
    print(f"\nOverall log-loss: {ll:.4f}")

    true_cls = true_labels.argmax(axis=1) if true_labels.ndim == 2 else true_labels
    pred_cls = pred_probs.argmax(axis=1)

    print("\nClassification Report:")
    print(classification_report(true_cls, pred_cls, target_names=CLASS_NAMES))

    plot_confusion_matrix(true_labels, pred_probs, output_path=f"{output_dir}/confusion_matrix.png")
    per_class = plot_per_class_log_loss(true_labels, pred_probs, output_path=f"{output_dir}/per_class_ll.png")

    return {"log_loss": ll, "per_class_log_loss": per_class.to_dict()}
