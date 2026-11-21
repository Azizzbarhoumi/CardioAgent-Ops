"""FastAPI application for uploaded heart-sound inference."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.settings import settings
from api.routers.analysis import router
from api.routers.samples import router as samples_router
from api.services.benchmark import BenchmarkService
from api.services.mlflow_logging import InferenceLogger
from audio_engine.inference.model_registry import ModelRegistry


def create_app(registry: ModelRegistry | None = None, inference_logger: InferenceLogger | None = None, app_settings=settings, benchmark: BenchmarkService | None = None) -> FastAPI:
    model_registry = registry or ModelRegistry(model_dir=app_settings.model_dir)
    logger = inference_logger or InferenceLogger(app_settings.mlflow_tracking_uri)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not model_registry.all_loaded:
            model_registry.load_all()
        yield

    app = FastAPI(
        title="CardioAgent PCG Inference API",
        description="Independent heart-sound model predictions for educational use.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.registry = model_registry
    app.state.inference_logger = logger
    app.state.settings = app_settings
    app.state.benchmark = benchmark or BenchmarkService(app_settings.mlflow_tracking_uri)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(samples_router)
    return app


app = create_app()
