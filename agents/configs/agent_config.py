"""Agent-orchestrator configuration: service URLs, timeouts, model names."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration for the agent orchestrator.

    Every field defaults from an environment variable so Docker Compose
    ``environment:`` blocks override them at deploy time.
    """

    # Service URLs (matching docker-compose service names)
    audio_engine_url: str = os.getenv("AUDIO_ENGINE_URL", "http://audio-engine:8003")
    vector_db_url: str = os.getenv("VECTOR_DB_URL", "http://vector-db:6333")
    llm_server_url: str = os.getenv("LLM_SERVER_URL", "http://llm-server:11434")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

    # LLM settings
    llm_model: str = os.getenv("LLM_MODEL", "cardioagent")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # Timeouts (seconds)
    audio_engine_timeout: float = float(os.getenv("AUDIO_ENGINE_TIMEOUT", "60"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))
    rag_timeout: float = float(os.getenv("RAG_TIMEOUT", "30"))


agent_config = AgentConfig()

