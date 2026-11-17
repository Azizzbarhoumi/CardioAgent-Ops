"""Analysis, health, and history endpoints."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from api.schemas.analysis import AnalysisResponse, HealthResponse, HistoryResponse
from api.services.analysis import analyze_recording
from audio_engine.features.audio_features import TABULAR_FEATURE_ORDER, CNN_FEATURE_SETTINGS, CNN_IMAGE_SIZE


router = APIRouter()


@router.get("/models")
def models_info(request: Request) -> dict:
    """Describe loaded models using the local provenance manifest."""
    registry = request.app.state.registry
    try:
        manifest_path = getattr(registry, "manifest_path", None)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else {}
    except (OSError, ValueError):
        manifest = {}
    metadata = {
        "svm_linear": {"type": "Support Vector Machine", "kernel": "linear", "input_features": list(TABULAR_FEATURE_ORDER), "score_type": "decision_score", "threshold": 0.0},
        "svm_rbf": {"type": "Support Vector Machine", "kernel": "RBF", "input_features": list(TABULAR_FEATURE_ORDER), "score_type": "decision_score", "threshold": 0.0},
        "decision_tree": {"type": "Decision Tree", "input_features": list(TABULAR_FEATURE_ORDER), "score_type": "probability", "threshold": 0.5},
        "cnn_custom": {"type": "Convolutional Neural Network", "input_shape": [*CNN_IMAGE_SIZE, 3], "mel_settings": CNN_FEATURE_SETTINGS, "score_type": "probability", "threshold": float(os.getenv("CNN_DECISION_THRESHOLD", "0.5"))},
    }
    result = []
    for key, details in metadata.items():
        entry = registry.models.get(key)
        info = manifest.get("models", {}).get(key, {})
        metrics = next((info[key] for key in ("reported_metrics", "reproduced_mean_metrics", "reproduced_metrics", "selected_fold_metrics") if info.get(key)), {})
        result.append({
            "key": key,
            "name": entry.name if entry else key,
            "loaded": entry is not None,
            "version": getattr(entry, "version", None) if entry else None,
            "model_run_id": getattr(entry, "run_id", None) if entry else None,
            "label_convention": {"0": "normal", "1": "abnormal"},
            "source": info.get("source", "Local model export"),
            "metrics": metrics,
            **details,
        })
    return {"models": result, "warning": "CNN uses the current runtime threshold of 0.5; no separate validation-selected threshold is recorded."}


@router.post("/analyze_pcg", response_model=AnalysisResponse)
async def analyze_pcg(request: Request, file: UploadFile = File(...)) -> AnalysisResponse:
    filename = file.filename or "recording.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=415, detail="Upload a .wav audio file.")
    wav_bytes = await file.read(request.app.state.settings.api_max_upload_bytes + 1)
    if len(wav_bytes) > request.app.state.settings.api_max_upload_bytes:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the configured size limit.")
    try:
        result = analyze_recording(
            filename=filename,
            wav_bytes=wav_bytes,
            registry=request.app.state.registry,
            logger=request.app.state.inference_logger,
            minimum_seconds=request.app.state.settings.min_audio_seconds,
            maximum_seconds=request.app.state.settings.max_audio_seconds,
        )
    except ValueError as exc:
        status_code = 422 if "too short" in str(exc).lower() or "too long" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AnalysisResponse(**result)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    registry = request.app.state.registry
    mlflow_reachable = request.app.state.inference_logger.reachable()
    return HealthResponse(
        status="ok" if registry.all_loaded else "degraded",
        models_loaded=registry.all_loaded,
        loaded_models=[entry.name for entry in registry.models.values()],
        model_load_errors=registry.load_errors,
        mlflow_reachable=mlflow_reachable,
    )


@router.get("/history", response_model=HistoryResponse)
def history(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> HistoryResponse:
    import mlflow

    try:
        mlflow.set_tracking_uri(request.app.state.settings.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name("inference_logs")
        if experiment is None:
            return HistoryResponse(analyses=[])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )
        analyses = []
        for _, row in runs.iterrows():
            metrics = {key.removeprefix("metrics."): float(value) for key, value in row.items() if key.startswith("metrics.") and value == value}
            start_time = row["start_time"]
            start_time_ms = int(start_time.timestamp() * 1000) if hasattr(start_time, "timestamp") else int(start_time)
            analyses.append({
                "run_id": row["run_id"],
                "run_name": row.get("tags.mlflow.runName"),
                "run_url": f"{os.getenv('MLFLOW_PUBLIC_URL', os.getenv('MLFLOW_UI_URI', request.app.state.settings.mlflow_tracking_uri)).rstrip('/')}/#/experiments/{experiment.experiment_id}/runs/{row['run_id']}",
                "experiment_id": experiment.experiment_id,
                "start_time": start_time_ms,
                "filename": row.get("params.filename"),
                "duration_s": row.get("params.duration_s"),
                "metrics": metrics,
                "tags": {key.removeprefix("tags."): value for key, value in row.items() if key.startswith("tags.") and value == value},
            })
        return HistoryResponse(analyses=analyses)
    except Exception as exc:
        return HistoryResponse(analyses=[], warning=f"Could not read MLflow history: {exc}")
