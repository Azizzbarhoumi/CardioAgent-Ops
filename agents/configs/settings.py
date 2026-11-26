"""Configuration settings for the 3-agent LangGraph orchestrator."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AgentConfig(BaseModel):
    """Runtime settings for AudioAgent, KnowledgeRAGAgent, and ReasoningLLMAgent."""

    audio_engine_url: str = Field(
        default_factory=lambda: os.getenv("AUDIO_ENGINE_URL", "http://localhost:8000").rstrip("/")
    )
    mock_audio: bool = Field(default_factory=lambda: _env_bool("MOCK_AUDIO", False))
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    )
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "cardio-qwen2.5-7b"))
    mock_llm: bool = Field(default_factory=lambda: _env_bool("MOCK_LLM", False))
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    http_timeout_s: float = Field(default=15.0, ge=0.5, le=120.0)
    rag_top_k: int = Field(default=3, ge=1, le=10)
