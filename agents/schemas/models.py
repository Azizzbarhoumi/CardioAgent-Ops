"""Pydantic v2 anti-hallucination schemas for inter-agent contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


EDUCATIONAL_DISCLAIMER = (
    "Educational clinical decision-support prototype only. Not a medical device; "
    "findings must be confirmed by a cardiologist and transthoracic echocardiography."
)


class AudioAnalysisResult(BaseModel):
    """Validated output from AudioAgent."""

    prediction: Literal["normal", "abnormal"]
    confidence: float = Field(ge=0.0, le=1.0)
    suspected_timing: Literal["systolic", "diastolic", "continuous", "normal", "unknown"] = "unknown"
    model_used: str = "audio-engine"
    source: Literal["audio-engine", "mock"] = "audio-engine"
    raw_details: dict[str, Any] = Field(default_factory=dict)


class RAGGuidelineChunk(BaseModel):
    """Single retrieved clinical guideline excerpt."""

    source: str
    section: str = ""
    score: float = Field(ge=0.0, le=1.0)
    content: str = Field(min_length=1)


class RAGContextResult(BaseModel):
    """Validated output from KnowledgeRAGAgent."""

    query: str = Field(min_length=1)
    retrieved_chunks: list[RAGGuidelineChunk] = Field(default_factory=list)
    summary: str = ""


class ClinicalReasoningReport(BaseModel):
    """Strict Pydantic v2 guardrail schema for ReasoningLLMAgent output."""

    summary: str = Field(min_length=10)
    risk_level: Literal["low", "moderate", "high"]
    suspected_conditions: list[str] = Field(default_factory=list)
    guideline_references: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(min_length=1)
    disclaimer: str = EDUCATIONAL_DISCLAIMER
    grounded: bool = True

    @field_validator("disclaimer")
    @classmethod
    def ensure_non_diagnostic_disclaimer(cls, value: str) -> str:
        if not value.strip():
            return EDUCATIONAL_DISCLAIMER
        return value
