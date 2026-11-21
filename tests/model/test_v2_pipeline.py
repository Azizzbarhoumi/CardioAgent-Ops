"""Phase 5 checks for the benchmark-v2 models, the API and the MLflow structure.

* golden snapshot: API output == training-side pipeline output (tests/fixtures/golden_v2)
* consistency: label rules, shapes, determinism, one preprocessing pass
* threshold protocol: selection refuses anything but the validation split
* MLflow (temporary local store, never the live server): parent/child structure,
  tags, figures, links, mean_* == average of the 3 children
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from audio_engine.evaluation.mlflow_logging import BENCHMARK_VERSION

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_v2"
MANIFEST = ROOT / "audio_engine" / "artifacts" / "models" / "model_manifest.json"
RESULTS = ROOT / "audio_engine" / "evaluation" / "results" / "v3"
MODEL_KEYS = ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom", "vgg16_transfer")

pytestmark = pytest.mark.skipif(not MANIFEST.is_file() or not (GOLDEN / "expected.json").is_file(), reason="v2 models not exported")


class _NoLog:
    def log(self, **_):
        return None, None, None

    def reachable(self):
        return False


class _FakeBenchmark:
    def summary_for(self, key):
        return {"benchmark_run_url": f"http://mlflow.test/#/experiments/1/runs/{key}", "benchmark_folds_urls": [],
                "benchmark_train_vs_test_url": None, "benchmark_summary": None, "benchmark_warning": None}

    def fetch(self, force=False):
        return {}


@pytest.fixture(scope="module")
def registry():
    from audio_engine.inference.model_registry import ModelRegistry

    reg = ModelRegistry(model_dir=MANIFEST.parent, manifest_path=MANIFEST)
    reg.load_all()
    assert reg.all_loaded, reg.load_errors
    return reg


@pytest.fixture(scope="module")
def client(registry):
    from fastapi.testclient import TestClient

    from api.main import create_app

    with TestClient(create_app(registry=registry, inference_logger=_NoLog(), benchmark=_FakeBenchmark())) as c:
        yield c


@pytest.fixture(scope="module")
def golden():
    return json.loads((GOLDEN / "expected.json").read_text())


# ------------------------------------------------------------------ golden snapshot
def test_golden_set_covers_both_classes_sources_and_lengths(golden):
    recs = golden["recordings"].values()
    assert len(golden["recordings"]) >= 6
    assert {r["true_label"] for r in recs} == {0, 1}
    assert {r["source"] for r in recs} == {"pascal", "physio"}
    assert min(r["duration_s"] for r in recs) < 2 and max(r["duration_s"] for r in recs) > 60


@pytest.mark.parametrize("wav_name", sorted(p.name for p in (GOLDEN / "wavs").glob("*.wav")))
def test_api_reproduces_golden_snapshot(client, golden, wav_name):
    expected = golden["recordings"][wav_name]["models"]
    response = client.post("/analyze_pcg", files={"file": (wav_name, (GOLDEN / "wavs" / wav_name).read_bytes(), "audio/wav")})
    assert response.status_code == 200, response.text
    got = {r["model_key"]: r for r in response.json()["results"]}
    assert set(got) == set(MODEL_KEYS)
    for key, exp in expected.items():
        tol = golden["tolerance"][exp["score_type"]]
        assert got[key]["prediction"] == exp["prediction"], (key, got[key]["score"], exp["score"])
        assert got[key]["score"] == pytest.approx(exp["score"], abs=tol), key
        assert got[key]["threshold"] == pytest.approx(exp["threshold"], abs=1e-12)
        assert got[key]["score_type"] == exp["score_type"]


# ------------------------------------------------------------------ consistency
def test_label_rule_matches_saved_threshold(client):
    manifest = json.loads(MANIFEST.read_text())["models"]
    for wav in sorted((GOLDEN / "wavs").glob("*.wav"))[:3]:
        for r in client.post("/analyze_pcg", files={"file": (wav.name, wav.read_bytes(), "audio/wav")}).json()["results"]:
            assert r["prediction"] in (0, 1) and r["prediction_label"] == {0: "Normal", 1: "Abnormal"}[r["prediction"]]
            assert r["threshold"] == manifest[r["model_key"]]["threshold"]
            if r["score_type"] == "decision_score":
                assert r["model_key"].startswith("svm_") and r["threshold"] == 0.0
                assert r["prediction"] == int(r["score"] > 0)
            else:
                assert 0.0 <= r["score"] <= 1.0
                assert r["prediction"] == int(r["score"] >= r["threshold"])


def test_inputs_have_training_shapes_and_feature_order(registry):
    from audio_engine.features.audio_features import TABULAR_FEATURE_ORDER, cnn_v2_input, extract_gray_mel, vgg_input
    from audio_engine.preprocessing.audio import decode_wav

    audio = decode_wav(next((GOLDEN / "wavs").glob("physio*.wav")).read_bytes())
    gray, _ = extract_gray_mel(audio.signal, audio.sample_rate)
    assert gray.dtype == np.uint8 and gray.shape[0] == 40
    assert cnn_v2_input(gray).shape == (40, 256, 1)
    assert vgg_input(gray).shape == (224, 224, 3)
    assert TABULAR_FEATURE_ORDER[:2] == ("duration", "snr") and len(TABULAR_FEATURE_ORDER) == 22
    for key in ("svm_linear", "svm_rbf"):
        assert registry.models[key].model.n_features_in_ == 22
    assert registry.models["decision_tree"].model.n_features_in_ == 22
    assert tuple(registry.models["cnn_custom"].model.input_shape[1:]) == (40, 256, 1)
    assert tuple(registry.models["vgg16_transfer"].model.input_shape[1:]) == (224, 224, 3)


def test_predictions_are_deterministic(registry):
    wav = next((GOLDEN / "wavs").glob("pascal*.wav")).read_bytes()
    a = registry.predict_all(wav)["results"]
    b = registry.predict_all(wav)["results"]
    assert [(r["prediction"], r["score"]) for r in a] == [(r["prediction"], r["score"]) for r in b]


def test_one_preprocessing_pass_shared_by_all_models(registry, monkeypatch):
    import audio_engine.inference.model_registry as mr

    calls = {"decode": 0, "tabular": 0, "gray": 0}
    for name, key in (("decode_wav", "decode"), ("extract_tabular_features", "tabular"), ("extract_gray_mel", "gray")):
        original = getattr(mr, name)

        def counted(*args, _o=original, _k=key, **kwargs):
            calls[_k] += 1
            return _o(*args, **kwargs)

        monkeypatch.setattr(mr, name, counted)
    registry.predict_all(next((GOLDEN / "wavs").glob("*.wav")).read_bytes())
    assert calls == {"decode": 1, "tabular": 1, "gray": 1}


def test_no_combined_verdict_in_response(client):
    wav = next((GOLDEN / "wavs").glob("*.wav"))
    body = client.post("/analyze_pcg", files={"file": (wav.name, wav.read_bytes(), "audio/wav")}).json()
    text = json.dumps({k: v for k, v in body.items() if k not in ("mel_spectrogram_png_base64", "waveform_png_base64")}).lower()
    for forbidden in ("combined", "ensemble", "majority", "final_verdict", "overall_prediction"):
        assert forbidden not in text
    assert [g["key"] for g in body["model_groups"]] == ["svm", "decision_tree", "cnn", "transfer_learning"]


# ------------------------------------------------------------------ threshold protocol
def test_threshold_can_only_be_chosen_on_validation():
    from audio_engine.evaluation.metrics import choose_threshold

    y, s = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    assert 0.4 < choose_threshold(y, s, split_name="validation") <= 0.6
    for split in ("test", "train"):
        with pytest.raises(AssertionError):
            choose_threshold(y, s, split_name=split)


def test_saved_results_used_the_shared_manifests():
    from audio_engine.evaluation.folds import manifest_sha

    sha = manifest_sha()
    for key in MODEL_KEYS:
        payload = json.loads((RESULTS / f"{key}.json").read_text())
        assert payload["split_manifest_sha"] == sha, key
        assert [f["fold"] for f in payload["folds"]] == [1, 2, 3]
        for f in payload["folds"]:
            assert "choose_threshold" not in f.get("threshold_rule", "") or "validation" in f["threshold_rule"]
    assert json.loads(MANIFEST.read_text())["split_manifest_sha"] == sha


# ------------------------------------------------------------------ MLflow (temporary store)
@pytest.fixture(scope="module")
def temp_mlflow(tmp_path_factory, monkeypatch_module):
    import mlflow

    from audio_engine.evaluation.mlflow_logging import log_benchmark

    store = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    monkeypatch_module.setenv("MLFLOW_ARTIFACT_ROOT", str(store))
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("audio_benchmark", artifact_location=(store / "artifacts").as_uri())
    logged = {}
    for key in MODEL_KEYS:
        payload = json.loads((RESULTS / f"{key}.json").read_text())
        for fold in payload["folds"]:  # skip the large model files in this temporary store
            fold.pop("model_path", None)
            fold.pop("full_model_path", None)
        logged[key] = log_benchmark(payload, tracking_uri=uri)
    return uri, logged


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_mlflow_parent_child_structure_and_means(temp_mlflow):
    from mlflow import MlflowClient

    from audio_engine.evaluation.metrics import METRIC_NAMES

    uri, logged = temp_mlflow
    client = MlflowClient(uri)
    eid = client.get_experiment_by_name("audio_benchmark").experiment_id
    for key, ids in logged.items():
        parent = client.get_run(ids["parent_run_id"])
        tags = parent.data.tags
        assert tags["evaluation"] == "patient_grouped_3_fold_cv" and tags["benchmark_version"] == BENCHMARK_VERSION
        assert tags["selected_for_app"] == "true" and tags["split_manifest_sha"]
        assert (tags.get("kernel") is not None) == key.startswith("svm_")
        children = client.search_runs([eid], f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
        assert sorted(c.info.run_name for c in children) == [f"{key}_fold_{i}" for i in (1, 2, 3)]
        artifacts = {a.path for a in client.list_artifacts(parent.info.run_id, "figures")} | {a.path for a in client.list_artifacts(parent.info.run_id, "reports")}
        assert {"figures/train_vs_test.png", "figures/folds_overview.png", "reports/cv_summary.json", "reports/confusion_matrices.json"} <= artifacts
        for child in children:
            for split in ("train", "test"):
                assert all(f"{split}_{m}" in child.data.metrics for m in METRIC_NAMES)
            cm = {a.path for a in client.list_artifacts(child.info.run_id, "confusion_matrices")}
            assert {"confusion_matrices/train_confusion_matrix.json", "confusion_matrices/test_confusion_matrix.json"} <= cm
        for split in ("train", "test"):
            for m in METRIC_NAMES:
                independent = float(np.mean([c.data.metrics[f"{split}_{m}"] for c in children]))
                assert parent.data.metrics[f"mean_{split}_{m}"] == pytest.approx(independent, abs=1e-12), (key, split, m)
                assert parent.data.metrics[f"std_{split}_{m}"] == pytest.approx(float(np.std([c.data.metrics[f"{split}_{m}"] for c in children])), abs=1e-12)
        if key in ("cnn_custom", "vgg16_transfer"):
            assert len(client.get_metric_history(children[0].info.run_id, "epoch_validation_loss")) > 1


def test_benchmark_links_resolve_to_the_selected_runs(temp_mlflow, monkeypatch):
    from api.services.benchmark import BenchmarkService

    uri, logged = temp_mlflow
    monkeypatch.setenv("MLFLOW_PUBLIC_URL", "http://browser.example:5000")
    data = BenchmarkService(uri).fetch(force=True)
    assert set(data) == set(MODEL_KEYS)
    for key, ids in logged.items():
        item = data[key]
        assert item["benchmark_run_url"].startswith("http://browser.example:5000/#/experiments/")
        assert item["benchmark_run_url"].endswith(f"/runs/{ids['parent_run_id']}")
        fold_ids = [re.search(r"/runs/([0-9a-f]+)$", u).group(1) for u in item["benchmark_folds_urls"]]
        assert fold_ids == [ids["child_run_ids"][n] for n in (1, 2, 3)]
        assert item["train_vs_test_url"].endswith(f"/runs/{ids['parent_run_id']}/artifacts/figures/train_vs_test.png")
        assert item["mean"]["test"]["recall"] == pytest.approx(np.mean([f["test"]["recall"] for f in item["folds"]]), abs=1e-12)
