"""Shared LangGraph state definition passed across the 3-agent pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class CardioAgentState(TypedDict, total=False):
    """Shared state dictionary for AudioAgent -> KnowledgeRAGAgent -> ReasoningLLMAgent."""

    filename: str
    wav_path: str
    wav_bytes: bytes
    patient_context: str
    mock_audio: bool
    audio_result: dict[str, Any]
    rag_context: dict[str, Any]
    reasoning_report: dict[str, Any]
    trace: list[dict[str, Any]]
    errors: list[str]
