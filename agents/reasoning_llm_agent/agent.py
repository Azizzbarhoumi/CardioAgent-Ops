"""ReasoningLLM-Agent — generates structured clinical reasoning via the LLM server.

This is the third and final node in the LangGraph pipeline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents.prompts.templates import REASONING_AGENT_PROMPT, REASONING_AGENT_SYSTEM
from agents.schemas.output_schemas import ClinicalReasoning
from agents.state.agent_state import AgentState
from agents.tools.http_clients import call_llm_generate


async def reasoning_agent_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: ReasoningLLM-Agent.

    Reads audio analysis and RAG context from state, constructs a
    detailed prompt, calls the LLM server, validates the structured
    JSON output against the ClinicalReasoning Pydantic schema, and
    writes the final reasoning fields.
    """
    errors: list[str] = list(state.get("errors", []))

    # Build the prompt from state
    analysis = state.get("audio_analysis", {})
    prompt = REASONING_AGENT_PROMPT.format(
        filename=state.get("filename", "recording.wav"),
        prediction_label=state.get("audio_prediction_label", "Unknown"),
        confidence=state.get("audio_confidence", 0.0),
        model_name=state.get("audio_model_name", "Unknown"),
        duration_s=analysis.get("duration_s", 0),
        sample_rate=analysis.get("sample_rate_after_resampling", 2000),
        user_prompt=state.get("user_prompt", "Analyse this heart sound recording."),
        rag_context=state.get("rag_context", "(No guideline context available.)"),
    )

    try:
        raw_response = await call_llm_generate(
            prompt=prompt,
            system=REASONING_AGENT_SYSTEM,
        )
    except Exception as exc:
        errors.append(f"ReasoningLLM-Agent: LLM call failed: {exc}")
        return {"errors": errors}

    # Parse and validate JSON response
    try:
        # Strip potential markdown code fences
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        validated = ClinicalReasoning(**parsed)
    except (json.JSONDecodeError, Exception) as exc:
        errors.append(f"ReasoningLLM-Agent: output validation failed: {exc}")
        # Store the raw response even if validation failed
        return {
            "llm_reasoning": raw_response,
            "clinical_summary": "",
            "recommendations": [],
            "confidence_assessment": "",
            "errors": errors,
        }

    return {
        "llm_reasoning": validated.reasoning,
        "clinical_summary": validated.clinical_summary,
        "recommendations": validated.recommendations,
        "confidence_assessment": validated.confidence_assessment,
        "errors": errors,
    }


def reasoning_agent_sync(state: AgentState) -> dict[str, Any]:
    """Synchronous wrapper for LangGraph."""
    return asyncio.run(reasoning_agent_node(state))

