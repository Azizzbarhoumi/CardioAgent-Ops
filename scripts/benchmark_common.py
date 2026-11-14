"""Shared data-loading helpers for the reproducible audio benchmarks."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["duration", "snr", *[f"mel_{index}" for index in range(1, 21)]]
CLASS_NAMES = ["normal", "abnormal"]


def load_heart_sound_features(project_root: Path = PROJECT_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Load the SVM-compatible 22-column feature matrix and binary labels."""
    feature_files = {
        "normal": project_root / "features_normal_train.data",
        "abnormal": project_root / "features_abnormal_train.data",
    }
    missing = [str(path) for path in feature_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pre-extracted features are required. Missing: " + ", ".join(missing)
        )

    loaded: dict[str, list[list[float]]] = {}
    for class_name, path in feature_files.items():
        with path.open("rb") as file:
            loaded[class_name] = pickle.load(file)

    features = np.asarray(loaded["normal"] + loaded["abnormal"], dtype=np.float64)
    labels = np.asarray(
        [0] * len(loaded["normal"]) + [1] * len(loaded["abnormal"]), dtype=np.int32
    )
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected a 2D matrix with {len(FEATURE_NAMES)} features, got {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return features, labels
