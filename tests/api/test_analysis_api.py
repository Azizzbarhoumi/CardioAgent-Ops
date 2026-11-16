"""HTTP endpoint checks with inference mocked and a temporary MLflow store."""

from __future__ import annotations

from io import BytesIO

import mlflow
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from api.core.settings import Settings
from api.main import create_app
from api.services.mlflow_logging import InferenceLogger
from audio_engine.preprocessing.audio import decode_wav


def synthetic_wav(duration_s: float = 1.0) -> bytes:
    rate = 2000
    time = np.arange(int(rate * duration_s)) / rate
    signal = 0.15 * np.sin(2 * np.pi * 80 * time)
    buffer = BytesIO()
    sf.write(buffer, signal, rate, format="WAV")
    return buffer.getvalue()


class FakeRegistry:
    models = {key: type("Entry", (), {"name": key})() for key in ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom")}
    load_errors = {}
    all_loaded = True

    def predict_all(self, audio):
        assert hasattr(audio, "signal")  # audio was decoded just once by the API service
        keys = ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom")
        results = [
            {
                "model_name": key,
                "model_key": key,
                "model_version": "test-model",
                "model_run_id": "source-run",
                "prediction": index % 2,
                "prediction_label": "Abnormal" if index % 2 else "Normal",
                "confidence_or_score": 0.5,
                "confidence_type": "probability" if index >= 2 else "decision_score",
                "latency_ms": 1.0,
            }
            for index, key in enumerate(keys)
        ]
        return {"audio": audio, "results": results, "mel_png": b"mel image"}


class FakeLogger:
    def reachable(self) -> bool:
        return True

    def log(self, **kwargs):
        kwargs["response"]["mlflow_run_id"] = "analysis-run"
        return "analysis-run", "http://mlflow/#/experiments/2/runs/analysis-run", None


def make_client(max_upload_bytes: int = 4096) -> TestClient:
    app = create_app(
        registry=FakeRegistry(),
        inference_logger=FakeLogger(),
        app_settings=Settings(api_max_upload_bytes=max_upload_bytes, min_audio_seconds=0.25, max_audio_seconds=5),
    )
    return TestClient(app)


def test_analyze_happy_path_returns_separate_results() -> None:
    with make_client() as client:
        response = client.post("/analyze_pcg", files={"file": ("sample.wav", synthetic_wav(), "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["mlflow_run_id"] == "analysis-run"
    assert len(data["results"]) == 4
    assert [result["model_key"] for result in data["results"]] == ["svm_linear", "svm_rbf", "decision_tree", "cnn_custom"]
    assert data["sample_rate_after_resampling"] == 2000
    assert data["waveform_png_base64"]


def test_analyze_rejects_non_wav_extension() -> None:
    with make_client() as client:
        response = client.post("/analyze_pcg", files={"file": ("sample.mp3", b"not wav", "audio/mpeg")})
    assert response.status_code == 415
    assert ".wav" in response.json()["detail"]


def test_analyze_rejects_empty_and_corrupt_wav() -> None:
    with make_client() as client:
        empty = client.post("/analyze_pcg", files={"file": ("empty.wav", b"", "audio/wav")})
        corrupt = client.post("/analyze_pcg", files={"file": ("corrupt.wav", b"not a RIFF file", "audio/wav")})
    assert empty.status_code == 400
    assert corrupt.status_code == 400


def test_analyze_rejects_audio_outside_configured_duration_limits() -> None:
    with make_client(max_upload_bytes=100000) as client:
        short = client.post("/analyze_pcg", files={"file": ("short.wav", synthetic_wav(0.1), "audio/wav")})
        long = client.post("/analyze_pcg", files={"file": ("long.wav", synthetic_wav(5.5), "audio/wav")})
    assert short.status_code == 422
    assert "too short" in short.json()["detail"].lower()
    assert long.status_code == 422
    assert "too long" in long.json()["detail"].lower()


def test_analyze_rejects_upload_over_configured_size_limit() -> None:
    with make_client() as client:
        response = client.post("/analyze_pcg", files={"file": ("large.wav", b"x" * 4097, "audio/wav")})
    assert response.status_code == 413


def test_health_reports_models_and_mlflow() -> None:
    with make_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True
    assert response.json()["mlflow_reachable"] is True


def test_mlflow_analysis_logging_uses_local_file_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_PUBLIC_URL", "http://public-mlflow.example")
    assert __import__("os").environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "true"
    tracking_uri = tmp_path.as_uri()
    logger = InferenceLogger(tracking_uri)
    wav_bytes = synthetic_wav()
    audio = decode_wav(wav_bytes)
    response = {
        "filename": "local-test.wav",
        "duration_s": audio.duration_s,
        "sample_rate_after_resampling": audio.sample_rate,
        "results": [
            {
                "model_key": key,
                "model_name": key,
                "model_version": "version-test",
                "model_run_id": "source-test",
                "prediction": 0,
                "confidence_or_score": 0.25,
                "latency_ms": 1.2,
            }
            for key in ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom")
        ],
        "total_processing_ms": 12.0,
    }
    run_id, run_url, warning = logger.log(
        filename="local-test.wav",
        wav_bytes=wav_bytes,
        response=response,
        mel_png=b"mel",
        waveform_png=b"waveform",
    )
    assert warning is None
    assert run_id
    assert run_url.startswith("http://public-mlflow.example/#/")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("inference_logs")
    run = mlflow.get_run(run_id)
    assert experiment is not None
    assert run.data.params["filename"] == "local-test.wav"
    assert run.data.metrics["svm_linear.prediction"] == 0
    assert run.data.metrics["svm_linear.confidence_or_score"] == 0.25
    assert run.data.metrics["svm_linear.latency_ms"] == 1.2
    assert run.data.tags["svm_linear.version"] == "version-test"
    assert run.data.tags["svm_linear.run_id"] == "source-test"
    assert run.data.metrics["total_processing_ms"] == 12.0
    paths = {item.path for item in mlflow.tracking.MlflowClient().list_artifacts(run_id, "analysis")}
    assert {"analysis/uploaded.wav", "analysis/mel_spectrogram.png", "analysis/result.json", "analysis/waveform.png"} <= paths

    app = create_app(
        registry=FakeRegistry(),
        inference_logger=logger,
        app_settings=Settings(
            mlflow_tracking_uri=tracking_uri,
            api_max_upload_bytes=4096,
            min_audio_seconds=0.25,
            max_audio_seconds=5,
        ),
    )
    with TestClient(app) as client:
        history = client.get("/history?limit=5")
    assert history.status_code == 200
    payload = history.json()
    assert payload["analyses"] and payload["analyses"][0]["run_id"] == run_id, payload
    assert payload["analyses"][0]["run_url"].startswith("http://public-mlflow.example/#/")
