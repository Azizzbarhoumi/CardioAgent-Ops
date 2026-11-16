"""Analysis, health, and history endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from api.schemas.analysis import AnalysisResponse, HealthResponse, HistoryResponse
from api.services.analysis import analyze_recording


router = APIRouter()


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
