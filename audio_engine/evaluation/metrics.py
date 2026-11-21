"""One metric definition for every model (positive class = abnormal = 1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "specificity",
)
SPLITS = ("train", "validation", "test")


def compute_metrics(y_true, y_pred, score) -> dict:
    """``score`` is any number that grows with abnormality (probability or decision score)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, np.asarray(score, dtype=float))) if len(set(y_true)) == 2 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y_val, score_val, *, split_name: str) -> float:
    """Threshold maximizing balanced accuracy, ties broken toward 0.5.

    ``split_name`` must be ``"validation"``: this assertion keeps the test fold
    out of threshold selection.
    """
    if split_name != "validation":
        raise AssertionError(f"Threshold must be chosen on the validation split, not {split_name!r}")
    s = np.sort(np.unique(np.asarray(score_val, dtype=float)))
    mids = (s[:-1] + s[1:]) / 2 if len(s) > 1 else np.array([])
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], s, mids)))
    return float(max(candidates, key=lambda t: (balanced_accuracy_score(y_val, score_val >= t), -abs(t - 0.5))))


def summarize(fold_metrics: list[dict]) -> dict:
    """mean_<split>_<metric> / std_<split>_<metric> over folds (population std, ddof=0)."""
    out = {}
    for split in SPLITS:
        for name in METRIC_NAMES:
            vals = [m[split][name] for m in fold_metrics if split in m]
            if vals:
                out[f"mean_{split}_{name}"] = float(np.mean(vals))
                out[f"std_{split}_{name}"] = float(np.std(vals))
    return out
