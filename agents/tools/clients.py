"""HTTP client tools for communicating with audio-engine and llm-server."""

from __future__ import annotations

from typing import Any

import httpx


def post_audio_to_engine(
    *,
    audio_engine_url: str,
    filename: str,
    wav_bytes: bytes,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Send a WAV recording to audio-engine /analyze_pcg."""
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(
            f"{audio_engine_url.rstrip('/')}/analyze_pcg",
            files={"file": (filename, wav_bytes, "audio/wav")},
        )
        response.raise_for_status()
        return response.json()


def call_ollama_chat(
    *,
    ollama_base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float = 30.0,
) -> str:
    """Call Ollama /api/chat expecting JSON output."""
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(f"{ollama_base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
