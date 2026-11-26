"""Shared agent state — TypedDict for LangGraph StateGraph.

The state flows through: AudioAgent → KnowledgeRAG-Agent → ReasoningLLM-Agent.
Each agent reads from and writes to specific keys.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed through the LangGraph agent pipeline.

    Keys are grouped by which agent produces them.
    """

    # ── Input (set by the orchestrator before the graph runs) ──
    wav_bytes: bytes                    # raw .wav upload
    filename: str                       # original filename
    user_prompt: str                    # optional free-text clinical question

    # ── AudioAgent output ──
    audio_analysis: dict[str, Any]      # full audio-engine response
    audio_prediction_label: str         # "Normal" or "Abnormal"
    audio_confidence: float             # best model confidence
    audio_model_name: str               # name of the model used

    # ── KnowledgeRAG-Agent output ──
    rag_context: str                    # formatted guideline context
    rag_sources: list[dict[str, Any]]   # individual retrieval hits

    # ── ReasoningLLM-Agent output ──
    llm_reasoning: str                  # full LLM-generated reasoning text
    clinical_summary: str               # structured summary
    recommendations: list[str]          # actionable next steps
    confidence_assessment: str          # LLM self-assessed confidence

    # ── Pipeline metadata ──
    errors: list[str]                   # accumulated errors from any agent

