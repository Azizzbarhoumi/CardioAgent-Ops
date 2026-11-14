"""Exercise all exported models against one generated WAV input."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_engine.inference.model_registry import ModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_exported_models_load_and_predict_independently() -> None:
    model_dir = PROJECT_ROOT / "audio_engine" / "artifacts" / "models"
    required = ("svm_linear.joblib", "svm_rbf.joblib", "decision_tree.joblib", "cnn_custom.keras", "model_manifest.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        pytest.skip("Run scripts/export_inference_models.py first; missing: " + ", ".join(missing))

    rate = 2000
    time = np.arange(rate) / rate
    signal = 0.1 * np.sin(2 * np.pi * 90 * time)
    buffer = BytesIO()
    sf.write(buffer, signal, rate, format="WAV")

    registry = ModelRegistry(model_dir=model_dir)
    registry.load_all()
    assert registry.all_loaded, registry.load_errors
    analysis = registry.predict_all(buffer.getvalue())
    assert [result["model_key"] for result in analysis["results"]] == [
        "svm_linear", "svm_rbf", "decision_tree", "cnn_custom"
    ]
    assert all(result["prediction"] in (0, 1) for result in analysis["results"])
    assert all(result["latency_ms"] >= 0 for result in analysis["results"])
