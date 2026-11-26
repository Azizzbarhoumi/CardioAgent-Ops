"""Pydantic v2 output schemas — anti-hallucination guardrails.

Every agent's output is validated against one of these schemas before
the state is updated.  This ensures structured, type-safe data flows
and catches malformed LLM responses early.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── AudioAgent output schema ──────────────────────────────────────────


class AudioAnalysisResult(BaseModel):
    """Validated output of the AudioAgent."""

    prediction_label: str = Field(
        ...,
        pattern=r"^(Normal|Abnormal)$",
        description="Must be exactly 'Normal' or 'Abnormal'.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence score between 0 and 1.",
    )
    model_name: str = Field(
        ..., min_length=1,
        description="Name of the audio model that produced the prediction.",
    )
    duration_s: float = Field(
        ..., gt=0,
        description="Duration of the recording in seconds.",
    )
    sample_rate: int = Field(
        ..., gt=0,
        description="Sample rate after resampling.",
    )


# ── KnowledgeRAG-Agent output schema ──────────────────────────────────


class RAGSource(BaseModel):
    """A single RAG retrieval source."""

    text: str = Field(..., min_length=1, description="Chunk text content.")
    source: str = Field(..., min_length=1, description="Document source identifier.")
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score.")


class RAGOutput(BaseModel):
    """Validated output of the KnowledgeRAG-Agent."""

    context: str = Field(
        ..., min_length=1,
        description="Formatted context string for the LLM prompt.",
    )
    sources: list[RAGSource] = Field(
        default_factory=list,
        description="Individual retrieval hits with scores.",
    )
    num_sources: int = Field(
        ..., ge=0,
        description="Number of retrieved guideline chunks.",
    )


# ── ReasoningLLM-Agent output schema ──────────────────────────────────


class ClinicalReasoning(BaseModel):
    """Validated output of the ReasoningLLM-Agent.

    This is the **anti-hallucination** guardrail: the LLM must produce
    structured output matching this exact schema.
    """

    clinical_summary: str = Field(
        ..., min_length=10,
        description="Structured clinical summary of the analysis.",
    )
    reasoning: str = Field(
        ..., min_length=10,
        description="Step-by-step medical reasoning.",
    )
    recommendations: list[str] = Field(
        ..., min_length=1,
        description="At least one actionable clinical recommendation.",
    )
    confidence_assessment: str = Field(
        ...,
        pattern=r"^(high|medium|low)$",
        description="Self-assessed confidence: 'high', 'medium', or 'low'.",
    )
    guideline_references: list[str] = Field(
        default_factory=list,
        description="ESC guideline sources cited in the reasoning.",
    )
    disclaimer: str = Field(
        default="This analysis is for educational purposes only and is not a medical diagnosis.",
        description="Mandatory medical disclaimer.",
    )


# ── Full pipeline output (returned to API Gateway) ────────────────────


class PipelineOutput(BaseModel):
    """Complete validated output of the 3-agent pipeline."""

    audio_result: AudioAnalysisResult
    rag_output: RAGOutput
    clinical_reasoning: ClinicalReasoning
    errors: list[str] = Field(default_factory=list)

