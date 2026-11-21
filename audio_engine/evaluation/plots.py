"""Train-vs-test figures shared by all models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOT_METRICS = ("accuracy", "balanced_accuracy", "recall", "f1")


def train_vs_test(fold_metrics: list[dict], model_name: str, out_path: Path) -> Path:
    """Grouped bars: train vs test for each metric, per fold plus mean with std error bars."""
    labels = [f"Fold {m['fold']}" for m in fold_metrics] + ["Mean"]
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(4 * len(PLOT_METRICS), 3.6), sharey=True)
    x = np.arange(len(labels))
    for ax, metric in zip(axes, PLOT_METRICS):
        for offset, split, color in ((-0.2, "train", "#4C78A8"), (0.2, "test", "#F58518")):
            vals = [m[split][metric] for m in fold_metrics]
            heights = vals + [float(np.mean(vals))]
            errs = [0] * len(vals) + [float(np.std(vals))]
            ax.bar(x + offset, heights, 0.4, yerr=errs, capsize=3, label=split, color=color)
        ax.set_xticks(x, labels, rotation=30)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
    axes[0].legend()
    fig.suptitle(f"{model_name}: train vs test (patient-grouped 3-fold)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def folds_overview(fold_metrics: list[dict], model_name: str, out_path: Path) -> Path:
    """The 3 test confusion matrices side by side."""
    fig, axes = plt.subplots(1, len(fold_metrics), figsize=(3.6 * len(fold_metrics), 3.4))
    for ax, m in zip(np.atleast_1d(axes), fold_metrics):
        cm = np.array(m["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0, 1], ["normal", "abnormal"])
        ax.set_yticks([0, 1], ["normal", "abnormal"])
        ax.set_xlabel("predicted")
        ax.set_title(f"Fold {m['fold']} test\nrecall={m['test']['recall']:.3f}")
    fig.suptitle(model_name)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)
