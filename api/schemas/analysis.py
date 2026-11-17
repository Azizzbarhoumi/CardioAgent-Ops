"""Pydantic response models for PCG analysis and history."""

from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    model_name: str
    model_key: str
    model_version: str
    model_run_id: str | None = None
    prediction: int = Field(ge=0, le=1)
    prediction_label: str
    confidence_or_score: float
    confidence_type: str
    score: float | None = None
    score_type: str | None = None
    threshold: float | None = None
    latency_ms: float = Field(ge=0)
    kernel: str | None = None


class ModelGroup(BaseModel):
    key: str
    name: str
    models: list[ModelResult]


class AnalysisResponse(BaseModel):
    filename: str
    duration_s: float
    source_sample_rate: int
    sample_rate_after_resampling: int
    results: list[ModelResult]
    model_groups: list[ModelGroup] = Field(default_factory=list)
    waveform_points: list[tuple[float, float]] = Field(default_factory=list)
    total_processing_ms: float
    mel_spectrogram_png_base64: str
    waveform_png_base64: str
    mlflow_run_id: str | None = None
    mlflow_run_url: str | None = None
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    loaded_models: list[str]
    model_load_errors: dict[str, str]
    mlflow_reachable: bool


class HistoryResponse(BaseModel):
    analyses: list[dict]
    warning: str | None = None
