"""Log one model's 3-fold result as a parent run with 3 nested child runs (MLflow 3.x)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow

from audio_engine.evaluation.folds import PROJECT_ROOT, SPLITS_DIR
from audio_engine.evaluation.metrics import METRIC_NAMES
from audio_engine.evaluation.plots import folds_overview, train_vs_test

EXPERIMENT = "audio_benchmark"
BENCHMARK_VERSION = "v3"  # v3 = PhysioNet labels fixed (2026-11-21); v2 runs are label-inverted
EVALUATION_TAG = "patient_grouped_3_fold_cv"
SPLITS = ("train", "validation", "test")
HISTORY_KEYS = {"loss": "epoch_train_loss", "accuracy": "epoch_train_accuracy", "val_loss": "epoch_validation_loss",
                "val_accuracy": "epoch_validation_accuracy", "learning_rate": "epoch_learning_rate"}


def select_app_fold(payload: dict) -> dict:
    """The fold model the app serves: best validation balanced accuracy (never test)."""
    return max(payload["folds"], key=lambda f: (f["validation"]["balanced_accuracy"], f["validation"]["macro_f1"]))


def _flat(params: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in params.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": str(v)[:500]})
    return out


def _cm_text(cm) -> str:
    (tn, fp), (fn, tp) = cm
    return f"            pred normal  pred abnormal\ntrue normal   {tn:>9}  {fp:>13}\ntrue abnormal {fn:>9}  {tp:>13}\n"


def log_benchmark(payload: dict, *, tracking_uri: str | None = None, selected_for_app: bool = True) -> dict:
    """Return {"parent_run_id", "child_run_ids", "experiment_id"}."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT)
    key = payload["model"]
    app_fold = select_app_fold(payload)
    kernel = {"svm_linear": "linear", "svm_rbf": "rbf"}.get(key)
    tags = {"model": key, "display_name": payload["display_name"], "framework": payload["framework"],
            "evaluation": EVALUATION_TAG, "benchmark_version": BENCHMARK_VERSION,
            "split_manifest": SPLITS_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "split_manifest_sha": payload["split_manifest_sha"], "score_type": payload["score_type"],
            "label_convention": "0=normal,1=abnormal", "positive_class": "abnormal"}
    if kernel:
        tags["kernel"] = kernel
    child_ids = {}
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=key, tags=tags) as parent:
        tmp = Path(tmp)
        mlflow.set_tags({"selected_for_app": str(selected_for_app).lower(), "app_fold": str(app_fold["fold"]),
                         "app_threshold": str(app_fold["threshold"])})
        mlflow.log_params({**_flat(payload["params"]), "n_folds": len(payload["folds"])})
        mlflow.log_metrics(payload["summary"])
        for fold in payload["folds"]:
            n = fold["fold"]
            with mlflow.start_run(run_name=f"{key}_fold_{n}", nested=True,
                                  tags={**tags, "fold": str(n), "split_manifest_fold_sha": payload["fold_manifest_shas"][str(n)]
                                        if str(n) in payload["fold_manifest_shas"] else payload["fold_manifest_shas"][n]}) as child:
                child_ids[n] = child.info.run_id
                p = {"fold": n, "threshold": fold["threshold"], "threshold_rule": fold["threshold_rule"],
                     **{f"{s}_size": fold["sizes"][s] for s in SPLITS}}
                p.update({f"best.{k}": v for k, v in fold.get("best_params", {}).items()})
                mlflow.log_params({k: str(v) for k, v in p.items()})
                mlflow.log_metrics({f"{s}_{m}": fold[s][m] for s in SPLITS for m in METRIC_NAMES})
                mlflow.log_metric("threshold", fold["threshold"])
                for hk, name in HISTORY_KEYS.items():
                    for step, value in enumerate(fold.get("history", {}).get(hk, []), start=1):
                        mlflow.log_metric(name, value, step=step)
                for s in SPLITS:
                    mlflow.log_dict({"labels": ["normal", "abnormal"], "matrix": fold[s]["confusion_matrix"]},
                                    f"confusion_matrices/{s}_confusion_matrix.json")
                    mlflow.log_text(_cm_text(fold[s]["confusion_matrix"]), f"confusion_matrices/{s}_confusion_matrix.txt")
                mlflow.log_dict({k: v for k, v in fold.items() if k != "history"}, "reports/fold_report.json")
                for path_key in ("model_path", "full_model_path"):
                    if fold.get(path_key):
                        mlflow.log_artifact(str(PROJECT_ROOT / fold[path_key]), artifact_path="model")
        figures = [train_vs_test(payload["folds"], payload["display_name"], tmp / "train_vs_test.png"),
                   folds_overview(payload["folds"], payload["display_name"], tmp / "folds_overview.png")]
        for fig in figures:
            mlflow.log_artifact(str(fig), artifact_path="figures")
        summary = {k: v for k, v in payload.items() if k != "folds"}
        summary["folds"] = [{"fold": f["fold"], "threshold": f["threshold"], "sizes": f["sizes"],
                             **{s: {m: f[s][m] for m in METRIC_NAMES} for s in SPLITS}} for f in payload["folds"]]
        summary["child_run_ids"] = child_ids
        summary["app_fold"] = app_fold["fold"]
        mlflow.log_dict(summary, "reports/cv_summary.json")
        mlflow.log_dict({"labels": ["normal", "abnormal"],
                         "folds": [{"fold": f["fold"], **{s: f[s]["confusion_matrix"] for s in SPLITS}} for f in payload["folds"]]},
                        "reports/confusion_matrices.json")
        for f in SPLITS_DIR.glob("fold_*.csv"):
            mlflow.log_artifact(str(f), artifact_path="split_manifest")
        return {"parent_run_id": parent.info.run_id, "child_run_ids": child_ids, "experiment_id": experiment.experiment_id}
