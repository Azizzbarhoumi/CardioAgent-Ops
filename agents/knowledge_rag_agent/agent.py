"""KnowledgeRAG-Agent — retrieves ESC guideline context from the vector database.

This is the second node in the LangGraph pipeline.
"""

from __future__ import annotations

from typing import Any

from agents.prompts.templates import KNOWLEDGE_AGENT_QUERY_TEMPLATE
from agents.schemas.output_schemas import RAGOutput, RAGSource
from agents.state.agent_state import AgentState
from rag_engine.pipeline.rag_pipeline import RAGPipeline


def knowledge_agent_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: KnowledgeRAG-Agent.

    Reads the audio analysis output from state, formulates a retrieval
    query, queries the RAG pipeline, validates the output, and writes
    the RAG context and sources.
    """
    errors: list[str] = list(state.get("errors", []))

    prediction_label = state.get("audio_prediction_label", "Unknown")
    confidence = state.get("audio_confidence", 0.0)
    model_name = state.get("audio_model_name", "Unknown")
    user_prompt = state.get("user_prompt", "Analyse this heart sound recording.")

    # Build a focused retrieval query
    query = KNOWLEDGE_AGENT_QUERY_TEMPLATE.format(
        prediction_label=prediction_label,
        confidence=confidence,
        model_name=model_name,
        user_prompt=user_prompt,
    )

    try:
        pipeline = RAGPipeline()
        results = pipeline.query(query)
        context = pipeline.query_as_context(query)
    except Exception as exc:
        errors.append(f"KnowledgeRAG-Agent: RAG query failed: {exc}")
        return {
            "rag_context": "(RAG retrieval failed — no guideline context available.)",
            "rag_sources": [],
            "errors": errors,
        }

    # Validate through Pydantic schema
    try:
        sources = [
            RAGSource(text=r.text, source=r.source, score=r.score)
            for r in results
        ]
        validated = RAGOutput(
            context=context if context else "(No relevant guidelines found.)",
            sources=sources,
            num_sources=len(sources),
        )
    except Exception as exc:
        errors.append(f"KnowledgeRAG-Agent: schema validation failed: {exc}")
        return {
            "rag_context": context or "(Validation failed.)",
            "rag_sources": [],
            "errors": errors,
        }

    return {
        "rag_context": validated.context,
        "rag_sources": [s.model_dump() for s in validated.sources],
        "errors": errors,
    }

