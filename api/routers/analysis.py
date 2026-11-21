"""Analysis, health, and history endpoints."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from api.schemas.analysis import AnalysisResponse, HealthResponse, HistoryResponse
from api.services.analysis import analyze_recording


router = APIRouter()


@router.get("/models")
def models_info(request: Request) -> dict:
    """Describe the loaded models from the v2 manifest (score type, threshold, provenance)."""
    registry = request.app.state.registry
    manifest = getattr(registry, "manifest", {}) or {}
    result = []
    for key, info in manifest.get("models", {}).items():
        entry = registry.models.get(key)
        result.append({
            "key": key, "name": info.get("display_name", key), "loaded": entry is not None,
            "version": entry.version if entry else None, "model_run_id": info.get("training_run_id"),
            "benchmark_parent_run_id": info.get("benchmark_parent_run_id"),
            "label_convention": manifest.get("label_convention"), "framework": info.get("framework"),
            "input": info.get("input"), "feature_version": info.get("feature_version"),
            "score_type": info.get("score_type"), "threshold": entry.threshold if entry else info.get("threshold"),
            "label_rule": info.get("label_rule"), "kernel": info.get("kernel"), "app_fold": info.get("app_fold"),
            "metrics": info.get("cv_summary", {}),
        })
    return {"models": result, "benchmark_version": manifest.get("benchmark_version"),
            "split_manifest_sha": manifest.get("split_manifest_sha")}


@router.get("/benchmark")
def benchmark(request: Request) -> dict:
    """Per model: fold metrics (train/validation/test), mean, std and MLflow links."""
    try:
        data = request.app.state.benchmark.fetch()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow benchmark unavailable: {exc}") from exc
    order = ("svm_linear", "svm_rbf", "decision_tree", "cnn_custom", "vgg16_transfer")
    models = [data[k] for k in order if k in data] + [v for k, v in data.items() if k not in order]
    return {"experiment": "audio_benchmark", "evaluation": "patient_grouped_3_fold_cv", "models": models}


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
            benchmark=getattr(request.app.state, "benchmark", None),
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
