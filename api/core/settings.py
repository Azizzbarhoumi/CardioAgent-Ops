"""Environment-based API settings with local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    api_max_upload_bytes: int = int(os.getenv("API_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    min_audio_seconds: float = float(os.getenv("MIN_AUDIO_SECONDS", "0.25"))
    max_audio_seconds: float = float(os.getenv("MAX_AUDIO_SECONDS", "120"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parents[2] / "audio_engine" / "artifacts" / "models")))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    )


settings = Settings()
