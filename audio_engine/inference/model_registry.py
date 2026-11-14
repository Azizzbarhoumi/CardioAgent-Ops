"""Load exported models once and produce independent model predictions."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from audio_engine.features.audio_features import extract_cnn_image, extract_tabular_features
from audio_engine.preprocessing.audio import AudioInput, decode_wav


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "audio_engine" / "artifacts" / "models"
LABELS = {0: "Normal", 1: "Abnormal"}


@dataclass
class LoadedModel:
    key: str
    name: str
    version: str
    run_id: str | None
    model: object


class ModelRegistry:
    """Runtime registry for two SVM kernels, one tree, and one CNN."""

    def __init__(self, model_dir: Path | None = None, manifest_path: Path | None = None):
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()
        self.manifest_path = Path(manifest_path or os.getenv("MODEL_MANIFEST_PATH", self.model_dir / "model_manifest.json"))
        self.models: dict[str, LoadedModel] = {}
        self.load_errors: dict[str, str] = {}

    def load_all(self) -> None:
        """Load exported model files and retain their provenance from the manifest."""
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        specifications = {
            "svm_linear": ("SVM (linear)", "SVM_LINEAR_MODEL_PATH", "svm_linear.joblib", "joblib"),
            "svm_rbf": ("SVM (RBF)", "SVM_RBF_MODEL_PATH", "svm_rbf.joblib", "joblib"),
            "decision_tree": ("Decision Tree", "DECISION_TREE_MODEL_PATH", "decision_tree.joblib", "joblib"),
            "cnn_custom": ("CNN", "CNN_MODEL_PATH", "cnn_custom.keras", "tensorflow"),
        }
        for key, (display_name, env_name, default_filename, flavor) in specifications.items():
            try:
                path = Path(os.getenv(env_name, str(self.model_dir / default_filename))).expanduser().resolve()
                model_info = manifest.get("models", {}).get(key, {})
                model_uri = os.getenv(f"{key.upper()}_MLFLOW_MODEL_URI")
                if model_uri and flavor == "joblib":
                    import mlflow.sklearn

                    model = mlflow.sklearn.load_model(model_uri)
                elif path.is_file() and flavor == "joblib":
                    model = joblib.load(path)
                elif path.is_file():
                    import tensorflow as tf

                    model = tf.keras.models.load_model(path)
                else:
                    resolve_best = os.getenv("MLFLOW_RESOLVE_BEST_MODELS", "false").lower() in {"1", "true", "yes"}
                    if not resolve_best or key not in {"svm_linear", "svm_rbf"}:
                        raise FileNotFoundError(f"Model file not found: {path}")
                    from audio_engine.inference.mlflow_resolver import resolve_best_logged_model

                    resolved = resolve_best_logged_model(
                        f"svm_{key.removeprefix('svm_')}_model",
                        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                    )
                    import mlflow.sklearn

                    model = mlflow.sklearn.load_model(resolved["model_uri"])
                    model_info = {**model_info, "model_id": resolved["model_id"], "run_id": resolved["run_id"]}
                self.models[key] = LoadedModel(
                    key=key,
                    name=display_name,
                    version=str(model_info.get("model_id") or model_info.get("source") or "local"),
                    run_id=model_info.get("run_id"),
                    model=model,
                )
                self.load_errors.pop(key, None)
            except Exception as exc:  # keep the API health endpoint available for diagnosis
                self.load_errors[key] = str(exc)

    @property
    def all_loaded(self) -> bool:
        return len(self.models) == 4

    def predict_all(self, wav_bytes: bytes | AudioInput) -> dict:
        """Decode once, extract both training-compatible inputs once, predict independently."""
        audio = wav_bytes if isinstance(wav_bytes, AudioInput) else decode_wav(wav_bytes)
        features = extract_tabular_features(audio.signal, audio.sample_rate).reshape(1, -1)
        cnn_image, mel_png = extract_cnn_image(audio.signal, audio.sample_rate)
        results = []

        for key in ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom"):
            if key not in self.models:
                raise RuntimeError(f"Model {key} is not loaded: {self.load_errors.get(key, 'unknown error')}")
            entry = self.models[key]
            started = time.perf_counter()
            if key.startswith("svm_"):
                model = entry.model
                prediction = int(model.predict(features)[0])
                decision = float(np.asarray(model.decision_function(features)).reshape(-1)[0])
                confidence, confidence_type = decision, "decision_score"
            elif key == "decision_tree":
                model = entry.model
                prediction = int(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]
                classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["classifier"].classes_)
                confidence, confidence_type = float(probabilities[classes.index(prediction)]), "probability"
            else:
                probability_abnormal = float(np.asarray(entry.model.predict(cnn_image[np.newaxis, ...], verbose=0)).reshape(-1)[0])
                prediction = int(probability_abnormal >= 0.5)
                confidence = probability_abnormal if prediction == 1 else 1.0 - probability_abnormal
                confidence_type = "probability"
            latency_ms = (time.perf_counter() - started) * 1000.0
            results.append({
                "model_name": entry.name,
                "model_key": key,
                "model_version": entry.version,
                "model_run_id": entry.run_id,
                "prediction": prediction,
                "prediction_label": LABELS[prediction],
                "confidence_or_score": confidence,
                "confidence_type": confidence_type,
                "latency_ms": latency_ms,
            })
        return {"audio": audio, "tabular_features": features, "cnn_image": cnn_image, "mel_png": mel_png, "results": results}
