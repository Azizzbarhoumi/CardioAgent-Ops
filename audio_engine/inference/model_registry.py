"""Load the 5 benchmark-v2 models once and produce independent predictions.

Everything (file, score type, threshold, provenance) comes from ``model_manifest.json``,
written by ``scripts/export_app_models.py``. Labels: 0 = normal, 1 = abnormal.
There is no combined verdict: every model result stays separate.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from audio_engine.features.audio_features import cnn_v2_input, extract_gray_mel, extract_tabular_features, vgg_input
from audio_engine.preprocessing.audio import AudioInput, decode_wav

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "audio_engine" / "artifacts" / "models"
LABELS = {0: "Normal", 1: "Abnormal"}
MODEL_ORDER = ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom", "vgg16_transfer")
PATH_ENV = {
    "svm_linear": "SVM_LINEAR_MODEL_PATH", "svm_rbf": "SVM_RBF_MODEL_PATH", "decision_tree": "DECISION_TREE_MODEL_PATH",
    "cnn_custom": "CNN_MODEL_PATH", "vgg16_transfer": "TRANSFER_MODEL_PATH",
}
THRESHOLD_ENV = {"decision_tree": "DECISION_TREE_THRESHOLD", "cnn_custom": "CNN_DECISION_THRESHOLD", "vgg16_transfer": "TRANSFER_DECISION_THRESHOLD"}


@dataclass
class LoadedModel:
    key: str
    name: str
    version: str
    run_id: str | None
    model: object
    score_type: str = "probability"
    threshold: float = 0.5
    info: dict = field(default_factory=dict)


class ModelRegistry:
    """Runtime registry for SVM (linear, RBF), Decision Tree, CNN and VGG16 transfer learning."""

    def __init__(self, model_dir: Path | None = None, manifest_path: Path | None = None):
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()
        self.manifest_path = Path(manifest_path or os.getenv("MODEL_MANIFEST_PATH", self.model_dir / "model_manifest.json"))
        self.models: dict[str, LoadedModel] = {}
        self.load_errors: dict[str, str] = {}
        self.manifest: dict = {}

    def load_all(self) -> None:
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.manifest = {}
            self.load_errors["manifest"] = f"Cannot read {self.manifest_path}: {exc}"
        for key in MODEL_ORDER:
            try:
                info = self.manifest.get("models", {})[key]
                path = Path(os.getenv(PATH_ENV[key], str(self.model_dir / Path(info["file"]).name))).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Model file not found: {path}")
                if info["framework"] == "scikit-learn":
                    model = joblib.load(path)
                else:
                    import tensorflow as tf

                    model = tf.keras.models.load_model(path, compile=False)
                threshold = float(os.getenv(THRESHOLD_ENV[key], info["threshold"])) if key in THRESHOLD_ENV else float(info["threshold"])
                self.models[key] = LoadedModel(
                    key=key, name=info["display_name"], version=f"v2-fold{info['app_fold']}",
                    run_id=info.get("training_run_id"), model=model, score_type=info["score_type"],
                    threshold=threshold, info=info,
                )
                self.load_errors.pop(key, None)
            except Exception as exc:  # keep /health available for diagnosis
                self.load_errors[key] = str(exc)

    @property
    def all_loaded(self) -> bool:
        return len(self.models) == len(MODEL_ORDER)

    def predict_all(self, wav_bytes: bytes | AudioInput) -> dict:
        """Decode once, compute the tabular vector and the gray Mel image once, predict independently."""
        audio = wav_bytes if isinstance(wav_bytes, AudioInput) else decode_wav(wav_bytes)
        features = extract_tabular_features(audio.signal, audio.sample_rate).reshape(1, -1)
        gray, mel_db = extract_gray_mel(audio.signal, audio.sample_rate)
        image_inputs = {"cnn_custom": cnn_v2_input(gray), "vgg16_transfer": vgg_input(gray)}
        results = []
        for key in MODEL_ORDER:
            if key not in self.models:
                raise RuntimeError(f"Model {key} is not loaded: {self.load_errors.get(key, 'unknown error')}")
            entry = self.models[key]
            started = time.perf_counter()
            if key.startswith("svm_"):
                score = float(np.asarray(entry.model.decision_function(features)).reshape(-1)[0])
                prediction = int(score > entry.threshold)
                confidence = score
            elif key == "decision_tree":
                classes = list(entry.model.classes_)
                score = float(entry.model.predict_proba(features)[0][classes.index(1)])
                prediction = int(score >= entry.threshold)
                confidence = score if prediction else 1.0 - score
            else:
                score = float(np.asarray(entry.model.predict(image_inputs[key][np.newaxis, ...], verbose=0)).reshape(-1)[0])
                prediction = int(score >= entry.threshold)
                confidence = score if prediction else 1.0 - score
            results.append({
                "model_name": entry.name, "model_key": key, "model_version": entry.version, "model_run_id": entry.run_id,
                "prediction": prediction, "prediction_label": LABELS[prediction],
                "confidence_or_score": confidence, "confidence_type": entry.score_type,
                "score": score, "score_type": entry.score_type, "threshold": entry.threshold,
                "label_rule": entry.info.get("label_rule"),
                "kernel": entry.info.get("kernel"),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            })
        return {"audio": audio, "tabular_features": features, "gray_mel": gray, "mel_db": mel_db, "results": results}
