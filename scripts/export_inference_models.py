"""Export current MLflow SVM Logged Models and reproducible local models."""

from __future__ import annotations

import json
import os
import shutil
import csv
import filecmp
import ast
from pathlib import Path

import joblib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from benchmark_common import PROJECT_ROOT, load_heart_sound_features


MODEL_DIR = PROJECT_ROOT / "audio_engine" / "artifacts" / "models"
CNN_RUN_ID = "c17147e3bb564ff086fa332e581440c0"
LOCAL_TREE_RUN_ID = "a1550ba943f14761bc58dbff8e1768b2"


def metric_summary(X: np.ndarray, y: np.ndarray, estimator) -> dict[str, float]:
    fold_rows = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    for train_indices, test_indices in splitter.split(X, y):
        estimator.fit(X[train_indices], y[train_indices])
        prediction = estimator.predict(X[test_indices])
        fold_rows.append({
            "accuracy": accuracy_score(y[test_indices], prediction),
            "precision": precision_score(y[test_indices], prediction, zero_division=0),
            "recall": recall_score(y[test_indices], prediction, zero_division=0),
            "f1": f1_score(y[test_indices], prediction, zero_division=0),
        })
    return {f"mean_{key}": float(np.mean([row[key] for row in fold_rows])) for key in fold_rows[0]}


def svm_metric_summary(X: np.ndarray, y: np.ndarray, kernel: str) -> dict[str, float]:
    """Re-run the SVM notebook's unstratified KFold evaluation exactly."""
    fold_rows = []
    splitter = KFold(n_splits=3, shuffle=True, random_state=1)
    for train_indices, test_indices in splitter.split(X):
        estimator = make_pipeline(StandardScaler(), SVC(kernel=kernel, class_weight="balanced"))
        estimator.fit(X[train_indices], y[train_indices])
        prediction = estimator.predict(X[test_indices])
        fold_rows.append({
            "accuracy": accuracy_score(y[test_indices], prediction),
            "precision": precision_score(y[test_indices], prediction, zero_division=0),
            "recall": recall_score(y[test_indices], prediction, zero_division=0),
            "f1": f1_score(y[test_indices], prediction, zero_division=0),
        })
    return {f"mean_{key}": float(np.mean([row[key] for row in fold_rows])) for key in fold_rows[0]}


def metric_differences(actual: dict[str, float], expected: dict[str, float]) -> dict[str, float]:
    return {key: float(actual[key] - expected[key]) for key in actual if key in expected}


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("audio_benchmark")
    if experiment is None:
        raise RuntimeError("MLflow experiment 'audio_benchmark' was not found.")
    logged_models = mlflow.search_logged_models(experiment_ids=[experiment.experiment_id], output_format="list")
    exported: dict[str, dict] = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_heart_sound_features(PROJECT_ROOT)

    for kernel in ("linear", "rbf"):
        model_name = f"svm_{kernel}_model"
        candidates = [model for model in logged_models if model.name == model_name and model.status == "READY"]
        if not candidates:
            raise RuntimeError(f"No READY Logged Model named {model_name!r} was found.")
        logged_model = max(candidates, key=lambda item: item.creation_timestamp)
        model = mlflow.sklearn.load_model(f"models:/{logged_model.model_id}")
        model_path = MODEL_DIR / f"svm_{kernel}.joblib"
        joblib.dump(model, model_path)
        run = mlflow.get_run(logged_model.source_run_id)
        reported_metrics = {key: value for key, value in run.data.metrics.items() if key.startswith("mean_")}
        reproduced = svm_metric_summary(X, y, kernel)
        exported[f"svm_{kernel}"] = {
            "file": model_path.relative_to(PROJECT_ROOT).as_posix(),
            "source": "MLflow Logged Model",
            "source_tracking_uri": tracking_uri,
            "model_id": logged_model.model_id,
            "run_id": logged_model.source_run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "features": "duration, snr, mel_1..mel_20",
            "label_convention": {"0": "normal", "1": "abnormal"},
            "reported_metrics": reported_metrics,
            "reproduced_metrics": reproduced,
            "metric_differences": metric_differences(reproduced, reported_metrics),
        }

    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=1)
    tree_metrics = metric_summary(X, y, DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=1))
    tree.fit(X, y)
    tree_pipeline = Pipeline([("classifier", tree)])
    tree_path = MODEL_DIR / "decision_tree.joblib"
    joblib.dump(tree_pipeline, tree_path)
    tree_differences = {}
    local_db = PROJECT_ROOT / "mlflow.db"
    if local_db.is_file():
        local_client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{local_db.as_posix()}")
        try:
            local_tree_run = local_client.get_run(LOCAL_TREE_RUN_ID)
            expected_tree_metrics = {key: value for key, value in local_tree_run.data.metrics.items() if key.startswith("mean_")}
            tree_differences = metric_differences(tree_metrics, expected_tree_metrics)
        except Exception:
            pass
    exported["decision_tree"] = {
        "file": tree_path.relative_to(PROJECT_ROOT).as_posix(),
        "source": "course4-decisiontree.ipynb; refit on existing features using notebook hyperparameters",
        "source_tracking_uri": "sqlite:///mlflow.db (local project tracking store)",
        "run_id": LOCAL_TREE_RUN_ID,
        "features": "duration, snr, mel_1..mel_20",
        "label_convention": {"0": "normal", "1": "abnormal"},
        "hyperparameters": {"max_depth": 3, "class_weight": "balanced", "random_state": 1},
        "reproduced_metrics": tree_metrics,
        "metric_differences_vs_local_run": tree_differences,
    }

    cnn_source = PROJECT_ROOT / "mlartifacts" / "1" / CNN_RUN_ID / "artifacts" / "models" / "best_fold" / "best.keras"
    fold_model_root = PROJECT_ROOT / "audio_engine" / "artifacts" / "models" / "cnn_custom"
    if not cnn_source.is_file():
        cnn_source = fold_model_root / "fold_1" / "best.keras"
    if not cnn_source.is_file():
        raise FileNotFoundError("The existing trained TensorFlow CNN checkpoint could not be found.")
    matching_folds = [
        fold for fold in (1, 2, 3)
        if (fold_model_root / f"fold_{fold}" / "best.keras").is_file()
        and filecmp.cmp(cnn_source, fold_model_root / f"fold_{fold}" / "best.keras", shallow=False)
    ]
    if len(matching_folds) != 1:
        raise RuntimeError(f"Could not uniquely match the MLflow CNN artifact to a fold checkpoint: {matching_folds}")
    selected_cnn_fold = matching_folds[0]
    cnn_path = MODEL_DIR / "cnn_custom.keras"
    shutil.copy2(cnn_source, cnn_path)
    def decode_image(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (224, 224))
        return tf.cast(image, tf.float32) / 255.0, label
    cnn_fold_metrics = []
    cnn_fold_differences = {}
    for fold in (1, 2, 3):
        split_path = PROJECT_ROOT / "Data" / "processed" / "splits" / "cnn" / f"cnn_fold_{fold}.csv"
        test_paths, test_labels = [], []
        with split_path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                if row["split"] == "test":
                    test_paths.append(str(PROJECT_ROOT / row["relative_path"]))
                    test_labels.append(int(row["label"]))
        labels_array = np.asarray(test_labels, dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((test_paths, labels_array))
        dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)
        fold_model_path = fold_model_root / f"fold_{fold}" / "best.keras"
        fold_model = tf.keras.models.load_model(fold_model_path)
        fold_probabilities = fold_model.predict(dataset, verbose=0).reshape(-1)
        fold_predictions = (fold_probabilities >= 0.5).astype(np.int32)
        fold_metrics = {
            "accuracy": float(accuracy_score(labels_array, fold_predictions)),
            "precision": float(precision_score(labels_array, fold_predictions, zero_division=0)),
            "recall": float(recall_score(labels_array, fold_predictions, zero_division=0)),
            "f1": float(f1_score(labels_array, fold_predictions, zero_division=0)),
        }
        cnn_fold_metrics.append({"fold": fold, **fold_metrics})
        report_path = PROJECT_ROOT / "audio_engine" / "evaluation" / "cnn_custom" / f"fold_{fold}_report.txt"
        report_text = report_path.read_text(encoding="utf-8")
        expected = ast.literal_eval(report_text.split("Metrics:", 1)[1].split("\n", 1)[0])
        cnn_fold_differences[str(fold)] = {
            key: fold_metrics[key] - float(expected[key]) for key in ("accuracy", "precision", "recall")
        }
        del fold_model
        tf.keras.backend.clear_session()
    cnn_mean_metrics = {
        f"mean_{key}": float(np.mean([row[key] for row in cnn_fold_metrics]))
        for key in ("accuracy", "precision", "recall", "f1")
    }
    cnn_reported_metrics = {}
    cnn_mean_differences = {}
    local_db = PROJECT_ROOT / "mlflow.db"
    if local_db.is_file():
        local_client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{local_db.as_posix()}")
        try:
            cnn_run = local_client.get_run(CNN_RUN_ID)
            cnn_reported_metrics = {
                key: value for key, value in cnn_run.data.metrics.items()
                if key in {"mean_accuracy", "mean_precision", "mean_recall", "mean_f1"}
            }
            cnn_mean_differences = metric_differences(cnn_mean_metrics, cnn_reported_metrics)
        except Exception:
            pass
    selected_fold_metrics = next(row for row in cnn_fold_metrics if row["fold"] == selected_cnn_fold)
    exported["cnn_custom"] = {
        "file": cnn_path.relative_to(PROJECT_ROOT).as_posix(),
        "source": f"Existing TensorFlow CNN checkpoint selected by notebook validation recall/F1 (fold {selected_cnn_fold})",
        "source_tracking_uri": "sqlite:///mlflow.db (local project tracking store)",
        "run_id": CNN_RUN_ID,
        "input_shape": [224, 224, 3],
        "input_preprocessing": "40-band Mel, dB, per-record min-max, magma RGB, resize 224x224, divide by 255",
        "label_convention": {"0": "normal", "1": "abnormal"},
        "validation_split": f"Data/processed/splits/cnn/cnn_fold_{selected_cnn_fold}.csv",
        "selected_fold_metrics": selected_fold_metrics,
        "fold_metric_differences_vs_notebook": cnn_fold_differences,
        "reproduced_mean_metrics": cnn_mean_metrics,
        "reported_mean_metrics": cnn_reported_metrics,
        "mean_metric_differences": cnn_mean_differences,
    }

    manifest = {
        "experiment": "audio_benchmark",
        "tracking_uri": tracking_uri,
        "label_convention": {"0": "normal", "1": "abnormal"},
        "models": exported,
    }
    manifest_path = MODEL_DIR / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        name: model.get("reproduced_metrics", model.get("reproduced_mean_metrics", model.get("reported_metrics")))
        for name, model in exported.items()
    }, indent=2))
    print(f"Wrote model manifest: {manifest_path}")


if __name__ == "__main__":
    main()
