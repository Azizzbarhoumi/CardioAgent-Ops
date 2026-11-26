"""Lightweight Ollama inference client with latency tracking."""

from __future__ import annotations

import time
from typing import Any

import httpx


def query_ollama(
    prompt: str,
    *,
    model: str = "cardio-qwen2.5-7b",
    base_url: str = "http://localhost:11434",
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Send a prompt to the quantized Ollama LLM server and record latency."""
    started = time.perf_counter()
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        payload = response.json()
    return {
        "model": model,
        "response": payload.get("response", ""),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }
