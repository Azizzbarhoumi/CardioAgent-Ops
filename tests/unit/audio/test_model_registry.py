from __future__ import annotations

import numpy as np

from audio_engine.inference.model_registry import LoadedModel, ModelRegistry
from audio_engine.preprocessing.audio import AudioInput


def test_prediction_reuses_one_preprocessed_signal_and_feature_vector(monkeypatch) -> None:
    import audio_engine.inference.model_registry as registry_module

    signal = np.zeros(2000, dtype=np.float32)
    audio = AudioInput(signal, sample_rate=2000, source_sample_rate=4000, duration_s=1.0)
    feature_vector = np.arange(22, dtype=np.float64)
    cnn_image = np.zeros((224, 224, 3), dtype=np.float32)
    mel_db = np.zeros((40, 16), dtype=np.float32)
    feature_extractions = []
    image_extractions = []
    model_inputs = []
    cnn_inputs = []

    def extract_features(received_signal, sample_rate):
        feature_extractions.append((received_signal, sample_rate))
        return feature_vector

    def extract_image(received_signal, sample_rate, *, return_mel=False):
        image_extractions.append((received_signal, sample_rate))
        return cnn_image, b"mel-png", mel_db

    class TabularModel:
        classes_ = np.array([0, 1])

        def predict(self, values):
            model_inputs.append(values)
            return np.array([0])

        def decision_function(self, values):
            model_inputs.append(values)
            return np.array([-0.25])

        def predict_proba(self, values):
            model_inputs.append(values)
            return np.array([[0.75, 0.25]])

    class CnnModel:
        def predict(self, values, verbose=0):
            cnn_inputs.append(values)
            return np.array([[0.25]])

    monkeypatch.setattr(registry_module, "extract_tabular_features", extract_features)
    monkeypatch.setattr(registry_module, "extract_cnn_image", extract_image)
    registry = ModelRegistry()
    registry.models = {
        key: LoadedModel(key, key, "test-version", None, CnnModel() if key == "cnn_custom" else TabularModel())
        for key in ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom")
    }

    result = registry.predict_all(audio)

    assert len(feature_extractions) == 1
    assert feature_extractions[0][0] is signal
    assert feature_extractions[0][1] == 2000
    assert len(image_extractions) == 1
    assert image_extractions[0][0] is signal
    assert image_extractions[0][1] == 2000
    assert len(model_inputs) == 6
    assert all(values is result["tabular_features"] for values in model_inputs)
    assert len(cnn_inputs) == 1
    assert cnn_inputs[0].shape == (1, 224, 224, 3)
    assert result["mel_db"] is mel_db
