"""3-fold benchmark lookups in MLflow (experiment audio_benchmark, benchmark v2).

The run is found through the tag ``selected_for_app = 'true'``, never by "latest run".
URLs are built from MLFLOW_PUBLIC_URL (the browser's address), not the internal URI.
"""

from __future__ import annotations

import os
import threading
import time

EXPERIMENT = "audio_benchmark"
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "specificity")
SPLITS = ("train", "validation", "test")
CACHE_SECONDS = float(os.getenv("BENCHMARK_CACHE_SECONDS", "60"))


def public_base(tracking_uri: str) -> str:
    return os.getenv("MLFLOW_PUBLIC_URL", os.getenv("MLFLOW_UI_URI", tracking_uri)).rstrip("/")


def run_url(base: str, experiment_id: str, run_id: str) -> str:
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def artifact_url(base: str, experiment_id: str, run_id: str, path: str) -> str:
    return f"{run_url(base, experiment_id, run_id)}/artifacts/{path}"


def confusion_metrics(folds: list[dict], split: str = "test") -> dict:
    """Recall, specificity and balanced accuracy per fold from the logged confusion matrices, then mean and
    population std (ddof=0) over the folds. Rows = true class, columns = predicted (0 normal, 1 abnormal)."""
    import numpy as np

    per_fold = []
    for fold in folds:
        m = fold["confusion_matrices"][split]
        recall = m["tp"] / (m["tp"] + m["fn"])
        specificity = m["tn"] / (m["tn"] + m["fp"])
        per_fold.append({"fold": fold["fold"], "recall": recall, "specificity": specificity,
                         "balanced_accuracy": (recall + specificity) / 2})
    out: dict = {"split": split, "source": "confusion_matrices", "per_fold": per_fold}
    for key in ("recall", "specificity", "balanced_accuracy"):
        values = np.array([f[key] for f in per_fold], dtype=float)
        out[f"mean_{key}"] = float(values.mean())
        out[f"std_{key}"] = float(values.std(ddof=0))
    return out


class BenchmarkService:
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self._cache: tuple[float, dict] | None = None
        self._lock = threading.Lock()
        self._matrix_cache: dict = {}

    def _client(self):
        from mlflow import MlflowClient

        return MlflowClient(self.tracking_uri)

    def fetch(self, force: bool = False) -> dict:
        """{model_key: {...}} for every model whose v2 parent run has selected_for_app=true."""
        with self._lock:
            if not force and self._cache and time.time() - self._cache[0] < CACHE_SECONDS:
                return self._cache[1]
        client = self._client()
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment is None:
            raise LookupError(f"MLflow experiment {EXPERIMENT!r} not found")
        eid = experiment.experiment_id
        base = public_base(self.tracking_uri)
        parents = client.search_runs(
            [eid], "tags.selected_for_app = 'true' and tags.benchmark_version = 'v3'", max_results=50
        )
        out: dict = {}
        for parent in parents:
            key = parent.data.tags.get("model")
            if key in out:
                raise RuntimeError(f"More than one run has selected_for_app=true for model {key!r}")
            children = sorted(
                client.search_runs([eid], f"tags.mlflow.parentRunId = '{parent.info.run_id}'"),
                key=lambda r: int(r.data.tags.get("fold", 0)),
            )
            m = parent.data.metrics
            out[key] = {
                "model_key": key,
                "display_name": parent.data.tags.get("display_name", key),
                "kernel": parent.data.tags.get("kernel"),
                "experiment_id": eid,
                "parent_run_id": parent.info.run_id,
                "benchmark_run_url": run_url(base, eid, parent.info.run_id),
                "train_vs_test_url": artifact_url(base, eid, parent.info.run_id, "figures/train_vs_test.png"),
                "benchmark_folds_urls": [run_url(base, eid, c.info.run_id) for c in children],
                "split_manifest_sha": parent.data.tags.get("split_manifest_sha"),
                "app_fold": parent.data.tags.get("app_fold"),
                "mean": {s: {k: m.get(f"mean_{s}_{k}") for k in METRICS} for s in SPLITS},
                "std": {s: {k: m.get(f"std_{s}_{k}") for k in METRICS} for s in SPLITS},
                "folds": [
                    {"fold": int(c.data.tags.get("fold", i + 1)), "run_id": c.info.run_id,
                     "run_url": run_url(base, eid, c.info.run_id),
                     **{s: {k: c.data.metrics.get(f"{s}_{k}") for k in METRICS} for s in SPLITS}}
                    for i, c in enumerate(children)
                ],
            }
        self._attach_confusion_matrices(client, out)
        with self._lock:
            self._cache = (time.time(), out)
        return out

    # ---- confusion matrices (read-only, additive) -------------------------------------------------
    def _read_matrix(self, client, run_id: str, split: str) -> dict:
        """Read confusion_matrices/<split>_confusion_matrix.json of a fold run. Cached forever per run id
        (a logged artifact never changes)."""
        key = (run_id, split)
        if key in self._matrix_cache:
            return self._matrix_cache[key]
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = client.download_artifacts(run_id, f"confusion_matrices/{split}_confusion_matrix.json", tmp)
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
        labels = [str(v) for v in raw["labels"]]
        if labels != ["normal", "abnormal"]:
            raise ValueError(f"unexpected label order {labels} in run {run_id}")
        (tn, fp), (fn, tp) = raw["matrix"]
        value = {"labels": labels, "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                 "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "total": int(tn + fp + fn + tp)}
        self._matrix_cache[key] = value
        return value

    def _attach_confusion_matrices(self, client, out: dict) -> None:
        """Add fold['confusion_matrices'] = {'test': ..., 'train': ...} and metrics derived from them.
        If an artifact cannot be read, the field stays empty and a warning is set; the rest is unchanged."""
        for item in out.values():
            item["label_convention"] = {"0": "normal", "1": "abnormal", "rows": "true class", "columns": "predicted class"}
            item["confusion_matrix_warning"] = None
            try:
                for fold in item["folds"]:
                    fold["confusion_matrices"] = {s: self._read_matrix(client, fold["run_id"], s) for s in ("test", "train")}
                item["confusion_metrics"] = confusion_metrics(item["folds"])
            except Exception as exc:  # never fail the whole response
                for fold in item["folds"]:
                    fold["confusion_matrices"] = {}
                item["confusion_metrics"] = None
                item["confusion_matrix_warning"] = f"Confusion matrices unavailable: {exc}"

    def summary_for(self, key: str) -> dict:
        """Fields added to every model result. Empty values (and a reason) if MLflow is unreachable."""
        try:
            item = self.fetch().get(key)
        except Exception as exc:  # the prediction must still be returned
            return {"benchmark_run_url": None, "benchmark_folds_urls": [], "benchmark_summary": None,
                    "benchmark_train_vs_test_url": None, "benchmark_warning": f"MLflow benchmark unavailable: {exc}"}
        if not item:
            return {"benchmark_run_url": None, "benchmark_folds_urls": [], "benchmark_summary": None,
                    "benchmark_train_vs_test_url": None, "benchmark_warning": "No selected_for_app=true run for this model"}
        return {
            "benchmark_run_url": item["benchmark_run_url"],
            "benchmark_folds_urls": item["benchmark_folds_urls"],
            "benchmark_train_vs_test_url": item["train_vs_test_url"],
            "benchmark_summary": {
                "n_folds": len(item["folds"]),
                "mean_test_recall": item["mean"]["test"]["recall"], "std_test_recall": item["std"]["test"]["recall"],
                "mean_test_accuracy": item["mean"]["test"]["accuracy"], "std_test_accuracy": item["std"]["test"]["accuracy"],
                "parent_run_id": item["parent_run_id"],
                # read-only, computed from the confusion matrices logged in MLflow (None if unreadable)
                **({f"{stat}_test_{k}": (item.get("confusion_metrics") or {}).get(f"{stat}_{k}")
                    for stat in ("mean", "std") for k in ("specificity", "balanced_accuracy")}),
                "metrics_source": "mlflow_confusion_matrices" if item.get("confusion_metrics") else None,
                "app_fold": item.get("app_fold"),
            },
            "benchmark_warning": None,
        }
