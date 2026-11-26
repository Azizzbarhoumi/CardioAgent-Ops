"""AudioAgent — calls the audio-engine service and parses predictions.

This is the first node in the LangGraph pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents.schemas.output_schemas import AudioAnalysisResult
from agents.state.agent_state import AgentState
from agents.tools.http_clients import call_audio_engine


async def audio_agent_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: AudioAgent.

    Reads ``wav_bytes`` and ``filename`` from state, calls the audio-engine
    microservice, validates the response, and writes audio analysis fields.
    """
    errors: list[str] = list(state.get("errors", []))

    wav_bytes = state.get("wav_bytes")
    if not wav_bytes:
        errors.append("AudioAgent: No wav_bytes in state.")
        return {"errors": errors}

    filename = state.get("filename", "recording.wav")

    try:
        response = await call_audio_engine(wav_bytes, filename)
    except Exception as exc:
        errors.append(f"AudioAgent: audio-engine call failed: {exc}")
        return {"errors": errors}

    # Pick the result with highest confidence among all models
    results = response.get("results", [])
    if not results:
        errors.append("AudioAgent: audio-engine returned no model results.")
        return {"audio_analysis": response, "errors": errors}

    best = max(results, key=lambda r: abs(r.get("confidence_or_score", 0)))

    # Validate through Pydantic schema
    try:
        validated = AudioAnalysisResult(
            prediction_label=best["prediction_label"],
            confidence=abs(best["confidence_or_score"]),
            model_name=best["model_name"],
            duration_s=response.get("duration_s", 0),
            sample_rate=response.get("sample_rate_after_resampling", 2000),
        )
    except Exception as exc:
        errors.append(f"AudioAgent: schema validation failed: {exc}")
        return {"audio_analysis": response, "errors": errors}

    return {
        "audio_analysis": response,
        "audio_prediction_label": validated.prediction_label,
        "audio_confidence": validated.confidence,
        "audio_model_name": validated.model_name,
        "errors": errors,
    }


def audio_agent_sync(state: AgentState) -> dict[str, Any]:
    """Synchronous wrapper for LangGraph (which calls nodes synchronously)."""
    return asyncio.run(audio_agent_node(state))

