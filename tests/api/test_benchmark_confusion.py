"""Read-only confusion-matrix metrics added to /benchmark and to each model's benchmark_summary.

Unit part: confusion_metrics() on hand-made matrices.
Integration part (skipped when MLflow is unreachable): recompute recall / specificity / balanced accuracy
independently from the matrices logged in MLflow and compare with (a) the service output and (b) the
per-fold metrics that the training runs logged; check matrix totals against the fold manifests.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile

import pytest

from api.services.benchmark import BenchmarkService, confusion_metrics

TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _fold(n, tn, fp, fn, tp):
    m = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return {"fold": n, "confusion_matrices": {"test": m}}


def test_confusion_metrics_hand_made():
    out = confusion_metrics([_fold(1, 8, 2, 1, 9), _fold(2, 5, 5, 5, 5)])
    assert out["per_fold"][0]["specificity"] == pytest.approx(0.8)
    assert out["per_fold"][0]["recall"] == pytest.approx(0.9)
    assert out["per_fold"][0]["balanced_accuracy"] == pytest.approx(0.85)
    assert out["mean_specificity"] == pytest.approx(0.65)
    assert out["std_specificity"] == pytest.approx(0.15)  # population std over folds (ddof=0)


def _mlflow_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{TRACKING}/health", timeout=3)
        return True
    except Exception:
        return False


mlflow_only = pytest.mark.skipif(not _mlflow_up(), reason="MLflow server not reachable")


@pytest.fixture(scope="module")
def service_data():
    return BenchmarkService(TRACKING).fetch(force=True)


def _independent_matrix(run_id: str, split: str) -> list[list[int]]:
    from mlflow import MlflowClient
    with tempfile.TemporaryDirectory() as tmp:
        path = MlflowClient(TRACKING).download_artifacts(run_id, f"confusion_matrices/{split}_confusion_matrix.json", tmp)
        raw = json.load(open(path, encoding="utf-8"))
    assert raw["labels"] == ["normal", "abnormal"]
    return raw["matrix"]


@mlflow_only
def test_specificity_and_balanced_accuracy_match_logged_matrices(service_data):
    assert len(service_data) == 5
    for key, item in service_data.items():
        assert item["confusion_matrix_warning"] is None, key
        spec, bal = [], []
        for fold in item["folds"]:
            (tn, fp), (fn, tp) = _independent_matrix(fold["run_id"], "test")
            s, r = tn / (tn + fp), tp / (tp + fn)
            spec.append(s)
            bal.append((s + r) / 2)
            # the logged per-fold metrics agree with the matrix
            assert fold["test"]["specificity"] == pytest.approx(s, abs=1e-9), key
            assert fold["test"]["balanced_accuracy"] == pytest.approx((s + r) / 2, abs=1e-9), key
        cm = item["confusion_metrics"]
        assert cm["mean_specificity"] == pytest.approx(statistics.fmean(spec), abs=1e-12)
        assert cm["std_specificity"] == pytest.approx(statistics.pstdev(spec), abs=1e-12)
        assert cm["mean_balanced_accuracy"] == pytest.approx(statistics.fmean(bal), abs=1e-12)
        assert cm["std_balanced_accuracy"] == pytest.approx(statistics.pstdev(bal), abs=1e-12)
        # and they match the parent run's logged mean/std
        assert cm["mean_specificity"] == pytest.approx(item["mean"]["test"]["specificity"], abs=1e-9)
        assert cm["std_specificity"] == pytest.approx(item["std"]["test"]["specificity"], abs=1e-9)


@mlflow_only
def test_summary_for_carries_the_new_fields(service_data):
    service = BenchmarkService(TRACKING)
    service._cache = (1e18, service_data)  # reuse the fetched data
    summary = service.summary_for("svm_linear")["benchmark_summary"]
    for field in ("mean_test_specificity", "std_test_specificity", "mean_test_balanced_accuracy", "std_test_balanced_accuracy"):
        assert isinstance(summary[field], float), field
    assert summary["metrics_source"] == "mlflow_confusion_matrices"
    # unchanged fields keep their logged values
    assert summary["mean_test_recall"] == service_data["svm_linear"]["mean"]["test"]["recall"]


@mlflow_only
def test_matrix_totals_equal_fold_sizes(service_data):
    from audio_engine.evaluation.folds import load_folds, manifest_path
    if not manifest_path(1).is_file():
        pytest.skip("fold manifests not built")
    sizes = {f.number: {s: len(f.split(s)) for s in ("train", "test")} for f in load_folds()}
    for key, item in service_data.items():
        for fold in item["folds"]:
            for split in ("test", "train"):
                assert fold["confusion_matrices"][split]["total"] == sizes[fold["fold"]][split], (key, fold["fold"], split)


def test_unreachable_mlflow_gives_empty_matrices_and_warning(monkeypatch):
    service = BenchmarkService("http://127.0.0.1:9")
    data = {"m": {"folds": [{"fold": 1, "run_id": "x"}]}}

    class Broken:
        def download_artifacts(self, *a, **k):
            raise ConnectionError("down")

    service._attach_confusion_matrices(Broken(), data)
    assert data["m"]["folds"][0]["confusion_matrices"] == {}
    assert data["m"]["confusion_metrics"] is None
    assert "unavailable" in data["m"]["confusion_matrix_warning"]
