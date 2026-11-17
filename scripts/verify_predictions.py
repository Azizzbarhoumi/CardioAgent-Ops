"""Compare the HTTP inference API against the Phase 1 frozen outputs."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
import sys
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.core.settings import settings
from api.main import create_app
from api.services.mlflow_logging import InferenceLogger
from audio_engine.features.audio_features import (
    CNN_FEATURE_SETTINGS,
    CNN_IMAGE_SIZE,
    TABULAR_FEATURE_ORDER,
)
from audio_engine.inference.model_registry import ModelRegistry
from audio_engine.preprocessing.audio import decode_wav


GOLDEN = ROOT / "tests" / "fixtures" / "golden" / "current_predictions.json"
TOLERANCE = 1e-5
MODEL_KEYS = ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom")


def compare_value(expected: float, actual: float, tolerance: float = TOLERANCE) -> bool:
    return bool(np.isclose(expected, actual, rtol=tolerance, atol=tolerance))


def _analyze(client: TestClient, relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    response = client.post("/analyze_pcg", files={"file": (path.name, path.read_bytes(), "audio/wav")})
    response.raise_for_status()
    return response.json()


def main() -> int:
    snapshot = json.loads(GOLDEN.read_text(encoding="utf-8"))
    registry = ModelRegistry(model_dir=settings.model_dir)
    registry.load_all()
    if not registry.all_loaded:
        raise RuntimeError(f"Could not load the complete model set: {registry.load_errors}")
    with tempfile.TemporaryDirectory(prefix="cardio-mlflow-verify-") as temp_dir:
        store = Path(temp_dir) / "mlflow"
        tracking_uri = store.as_uri()
        app = create_app(
            registry=registry,
            inference_logger=InferenceLogger(tracking_uri),
            app_settings=replace(settings, mlflow_tracking_uri=tracking_uri),
        )
        table: list[tuple[str, str, str, str, bool]] = []
        failures: list[str] = []
        with TestClient(app) as client:
            if client.get("/health").json().get("models_loaded") is not True:
                raise AssertionError("API health did not report all models loaded")
            model_info = client.get("/models").json()
            if len(model_info.get("models", [])) != 4:
                raise AssertionError("/models did not describe all four runtime models")
            for model in model_info.get("models", []):
                if model["key"] in ("svm_linear", "svm_rbf", "decision_tree"):
                    if model.get("input_features") != list(TABULAR_FEATURE_ORDER):
                        failures.append(f"/models feature order is incorrect for {model['key']}")
                if model["key"] == "cnn_custom" and model.get("input_shape") != [*CNN_IMAGE_SIZE, 3]:
                    failures.append("/models CNN input shape is incorrect")
            for item in snapshot["files"]:
                relative = item["file"]
                output = _analyze(client, relative)
                if output["filename"] != Path(relative).name:
                    failures.append(f"{relative}: API filename does not match the uploaded file")
                if output["source_sample_rate"] != item["source_sample_rate"]:
                    failures.append(f"{relative}: source sample rate mismatch")
                if output["sample_rate_after_resampling"] != item["sample_rate_after_resampling"]:
                    failures.append(f"{relative}: resampled rate mismatch")
                if not compare_value(item["duration_after_resampling_s"], output["duration_s"]):
                    failures.append(f"{relative}: resampled duration mismatch")
                if output.get("warning"):
                    failures.append(f"{relative}: unexpected analysis warning: {output['warning']}")
                by_key = {result["model_key"]: result for result in output["results"]}
                if set(by_key) != set(MODEL_KEYS):
                    failures.append(f"{relative}: API model set is incorrect: {sorted(by_key)}")
                for key in MODEL_KEYS:
                    expected = item["models"][key]
                    actual = by_key[key]
                    checks = [
                        actual["prediction"] == expected["prediction"],
                        compare_value(expected["score"], actual["confidence_or_score"]),
                        compare_value(expected["threshold"], actual["threshold"], tolerance=0.0),
                    ]
                    if key in ("decision_tree", "cnn_custom"):
                        checks.extend([
                            actual["score_type"] == "probability",
                            compare_value(expected["probability_abnormal"], actual["score"]),
                        ])
                    else:
                        checks.extend([
                            actual["score_type"] == "decision_score",
                            compare_value(expected["score"], actual["score"]),
                        ])
                    ok = all(checks)
                    expected_score = (
                        expected["probability_abnormal"]
                        if key in ("decision_tree", "cnn_custom")
                        else expected["score"]
                    )
                    expected_text = f"{expected['prediction']}/{expected_score:.5f}"
                    got_text = f"{actual['prediction']}/{actual['score']:.5f}"
                    table.append((relative, key, expected_text, got_text, ok))
                    if not ok:
                        failures.append(f"{relative} {key}: expected={expected}, got={actual}")
                    # Each binary classifier's reported label must agree with its signed score/threshold.
                    if key.startswith("svm_") and (actual["prediction"] != int(actual["score"] > 0)):
                        failures.append(f"{relative} {key}: decision score and label disagree")
                    if key == "decision_tree" and (actual["prediction"] != int(actual["score"] >= actual["threshold"])):
                        failures.append(f"{relative} {key}: P(abnormal) and label disagree")
                    if key == "cnn_custom" and (actual["prediction"] != int(actual["score"] >= actual["threshold"])):
                        failures.append(f"{relative} {key}: P(abnormal) and label disagree")

                # Preserve the existing grouping contract, including the two independent SVM kernels.
                groups = output["model_groups"]
                if [(g["name"], len(g["models"])) for g in groups] != [("SVM (2 kernels)", 2), ("Decision Tree", 1), ("CNN", 1)]:
                    failures.append(f"{relative}: model groups are malformed")
                if len(output["waveform_points"]) < 2 or not output["mel_spectrogram_png_base64"]:
                    failures.append(f"{relative}: audio chart payload is incomplete")
                if output["mlflow_run_id"] and output["mlflow_run_url"]:
                    import mlflow

                    client_mlflow = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                    run = client_mlflow.get_run(output["mlflow_run_id"])
                    if output["mlflow_run_id"] not in output["mlflow_run_url"]:
                        failures.append(f"{relative}: public MLflow URL does not identify its run")
                    if run.data.params.get("filename") != Path(relative).name:
                        failures.append(f"{relative}: MLflow filename param missing")
                    if run.data.params.get("sample_rate_after_resampling") != str(output["sample_rate_after_resampling"]):
                        failures.append(f"{relative}: MLflow sample-rate param mismatch")
                    if not compare_value(float(run.data.params.get("duration_s", "nan")), output["duration_s"]):
                        failures.append(f"{relative}: MLflow duration param mismatch")
                    if run.data.tags.get("source") != "ui" or run.data.tags.get("label_convention") != "0=normal,1=abnormal":
                        failures.append(f"{relative}: MLflow analysis tags are incomplete")
                    if not compare_value(
                        run.data.metrics.get("total_processing_ms", float("nan")),
                        output["total_processing_ms"],
                    ):
                        failures.append(f"{relative}: MLflow total processing metric mismatch")
                    for key, actual in by_key.items():
                        for metric_name, response_value in (
                            ("prediction", actual["prediction"]),
                            ("confidence_or_score", actual["confidence_or_score"]),
                            ("score", actual["score"]),
                            ("threshold", actual["threshold"]),
                            ("latency_ms", actual["latency_ms"]),
                        ):
                            metric = run.data.metrics.get(f"{key}.{metric_name}")
                            if metric is None or not compare_value(response_value, metric):
                                failures.append(f"{relative} {key}: MLflow {metric_name} metric mismatch")
                        if run.data.tags.get(f"{key}.name") != actual["model_name"]:
                            failures.append(f"{relative} {key}: MLflow model tag missing")
                        if run.data.tags.get(f"{key}.version") != actual["model_version"]:
                            failures.append(f"{relative} {key}: MLflow version tag mismatch")
                        if run.data.tags.get(f"{key}.run_id") != (actual.get("model_run_id") or "local_export"):
                            failures.append(f"{relative} {key}: MLflow training-run tag mismatch")
                    artifacts = {Path(a.path).name for a in client_mlflow.list_artifacts(run.info.run_id, "analysis")}
                    required = {"uploaded.wav", "mel_spectrogram.png", "waveform.png", "result.json"}
                    if not required.issubset(artifacts):
                        failures.append(f"{relative}: MLflow artifacts missing: {required - artifacts}")
                else:
                    failures.append(
                        f"{relative}: temporary MLflow run or public URL was not created "
                        f"({output.get('warning')})"
                    )

            # Repeat the shortest case to verify deterministic labels and scores.
            repeated_file = snapshot["files"][0]
            first = _analyze(client, repeated_file["file"])
            second = _analyze(client, repeated_file["file"])
            first_map = {r["model_key"]: r for r in first["results"]}
            second_map = {r["model_key"]: r for r in second["results"]}
            for key in MODEL_KEYS:
                if first_map[key]["prediction"] != second_map[key]["prediction"] or not compare_value(first_map[key]["score"], second_map[key]["score"]):
                    failures.append(f"Determinism failure: {repeated_file['file']} {key}")

        # Verify the shared feature extraction contract and the CNN tensor dimensions.
        first_path = ROOT / snapshot["files"][0]["file"]
        audio = decode_wav(first_path.read_bytes(), minimum_seconds=settings.min_audio_seconds, maximum_seconds=settings.max_audio_seconds)
        prediction = registry.predict_all(audio)
        if tuple(TABULAR_FEATURE_ORDER) != ("duration", "snr", *(f"mel_{i}" for i in range(1, 21))):
            failures.append("Tabular feature names/order differ from the frozen contract")
        if prediction["tabular_features"].shape != (1, 22):
            failures.append(f"Tabular input shape is {prediction['tabular_features'].shape}, expected (1, 22)")
        if tuple(prediction["cnn_image"].shape) != (*CNN_IMAGE_SIZE, 3):
            failures.append(
                f"CNN input shape is {prediction['cnn_image'].shape}, expected {(*CNN_IMAGE_SIZE, 3)}"
            )
        if prediction["mel_db"].ndim != 2 or prediction["mel_db"].shape[0] != CNN_FEATURE_SETTINGS["n_mels"]:
            failures.append(
                f"CNN Mel shape is {prediction['mel_db'].shape}, expected "
                f"({CNN_FEATURE_SETTINGS['n_mels']}, frames)"
            )

    print(f"Tolerance for floating point scores: +/-{TOLERANCE:g}")
    print(f"{'File':64} {'Model':16} {'Expected label/score':22} {'Got label/score':22} Status")
    print("-" * 132)
    for filename, key, expected, got, ok in table:
        print(f"{filename[:64]:64} {key:16} {expected:22} {got:22} {'OK' if ok else 'FAIL'}")
    print(f"Checked {len(table)} file/model pairs, {len(snapshot['files'])} recordings, determinism, shape/order, and temporary MLflow logs.")
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        return 1
    print("All API predictions and checks match the Phase 1 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
