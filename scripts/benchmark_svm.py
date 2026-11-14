"""Run the heart-sound SVM benchmark and log it to MLflow."""

from __future__ import annotations

import argparse

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from benchmark_common import CLASS_NAMES, FEATURE_NAMES, load_heart_sound_features


def evaluate_kernel(X: np.ndarray, y: np.ndarray, kernel: str) -> tuple[list[dict], list[dict]]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    metrics, matrices = [], []
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        model = make_pipeline(StandardScaler(), SVC(kernel=kernel, class_weight="balanced"))
        model.fit(X[train_index], y[train_index])
        predicted = model.predict(X[test_index])
        metrics.append({
            "fold": fold,
            "accuracy": float(accuracy_score(y[test_index], predicted)),
            "precision": float(precision_score(y[test_index], predicted, zero_division=0)),
            "recall": float(recall_score(y[test_index], predicted, zero_division=0)),
            "f1": float(f1_score(y[test_index], predicted, zero_division=0)),
        })
        matrices.append({
            "fold": fold,
            "labels": CLASS_NAMES,
            "matrix": confusion_matrix(y[test_index], predicted, labels=[0, 1]).tolist(),
        })
    return metrics, matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    args = parser.parse_args()
    X, y = load_heart_sound_features()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("audio_benchmark")

    for kernel in ("linear", "rbf"):
        fold_metrics, matrices = evaluate_kernel(X, y, kernel)
        with mlflow.start_run(run_name=f"svm_{kernel}") as run:
            mlflow.set_tags({"model": "SVM", "task": "pcg_normal_vs_abnormal", "evaluation": "3_fold_cross_validation", "feature_set": "duration_snr_20_mel_means"})
            mlflow.log_params({"kernel": kernel, "class_weight": "balanced", "n_folds": 3, "n_features": X.shape[1], "n_samples": len(y), "features": "duration + snr + 20 mel means"})
            for metric in ("accuracy", "precision", "recall", "f1"):
                mlflow.log_metric(f"mean_{metric}", float(np.mean([row[metric] for row in fold_metrics])))
            for row in fold_metrics:
                for metric in ("accuracy", "precision", "recall", "f1"):
                    mlflow.log_metric(f"fold_{metric}", row[metric], step=row["fold"])
            mlflow.log_dict({"kernel": kernel, "folds": fold_metrics}, "reports/cv_metrics.json")
            mlflow.log_dict({"kernel": kernel, "folds": matrices}, "reports/confusion_matrices.json")
            model = make_pipeline(StandardScaler(), SVC(kernel=kernel, class_weight="balanced"))
            model.fit(X, y)
            mlflow.sklearn.log_model(model, name=f"svm_{kernel}_model", serialization_format="cloudpickle", signature=infer_signature(X[:5], model.predict(X[:5])), input_example=X[:5])
            print(f"Logged SVM {kernel}: {run.info.run_id}")


if __name__ == "__main__":
    main()
