"""Best-effort MLflow logging for one uploaded recording per run."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"

import mlflow


class InferenceLogger:
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.ui_uri = os.getenv("MLFLOW_UI_URI", tracking_uri).rstrip("/")

    def reachable(self) -> bool:
        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            client.search_experiments(max_results=1)
            return True
        except Exception:
            return False

    def log(self, *, filename: str, wav_bytes: bytes, response: dict, mel_png: bytes, waveform_png: bytes) -> tuple[str | None, str | None, str | None]:
        run_id = None
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.set_experiment("inference_logs")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)[:80] or "recording"
            run_name = f"{safe_stem}_{timestamp}"
            with tempfile.TemporaryDirectory(prefix="pcg-analysis-") as temp_dir:
                temp_path = Path(temp_dir)
                audio_path = temp_path / "uploaded.wav"
                mel_path = temp_path / "mel_spectrogram.png"
                waveform_path = temp_path / "waveform.png"
                result_path = temp_path / "result.json"
                audio_path.write_bytes(wav_bytes)
                mel_path.write_bytes(mel_png)
                waveform_path.write_bytes(waveform_png)
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
                    run_id = run.info.run_id
                    response["mlflow_run_id"] = run.info.run_id
                    response["mlflow_run_url"] = f"{self.ui_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
                    mlflow.log_params({
                        "filename": filename[:500],
                        "duration_s": f"{response['duration_s']:.6f}",
                        "sample_rate_after_resampling": response["sample_rate_after_resampling"],
                    })
                    mlflow.set_tags({"source": "ui", "label_convention": "0=normal,1=abnormal"})
                    for result in response["results"]:
                        key = result["model_key"]
                        mlflow.log_metric(f"{key}.prediction", result["prediction"])
                        mlflow.log_metric(f"{key}.confidence_or_score", result["confidence_or_score"])
                        mlflow.log_metric(f"{key}.latency_ms", result["latency_ms"])
                        mlflow.set_tag(f"{key}.name", result["model_name"])
                        mlflow.set_tag(f"{key}.version", result["model_version"])
                        mlflow.set_tag(f"{key}.run_id", result.get("model_run_id") or "local_export")
                    mlflow.log_metric("total_processing_ms", response["total_processing_ms"])
                    result_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
                    mlflow.log_artifacts(str(temp_path), artifact_path="analysis")
                    return run.info.run_id, response["mlflow_run_url"], None
        except Exception as exc:
            if run_id:
                try:
                    mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri).set_terminated(
                        run_id, status="FAILED"
                    )
                except Exception:
                    pass
            return None, None, f"Predictions completed, but MLflow logging failed: {exc}"
