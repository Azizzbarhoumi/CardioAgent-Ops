"""LangGraph StateGraph — wires AudioAgent → KnowledgeRAG → ReasoningLLM.

This module defines the 3-node directed graph that forms the core of
the agent-orchestrator (Service 4).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agents.audio_agent.agent import audio_agent_sync
from agents.knowledge_rag_agent.agent import knowledge_agent_node
from agents.reasoning_llm_agent.agent import reasoning_agent_sync
from agents.state.agent_state import AgentState


def build_cardio_graph() -> StateGraph:
    """Construct and compile the CardioAgent-Ops LangGraph StateGraph.

    The graph is a simple 3-node linear pipeline::

        AudioAgent → KnowledgeRAG-Agent → ReasoningLLM-Agent → END

    Each node reads from and writes to the shared :class:`AgentState`.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ──
    graph.add_node("audio_agent", audio_agent_sync)
    graph.add_node("knowledge_rag_agent", knowledge_agent_node)
    graph.add_node("reasoning_llm_agent", reasoning_agent_sync)

    # ── Define edges (linear pipeline) ──
    graph.set_entry_point("audio_agent")
    graph.add_edge("audio_agent", "knowledge_rag_agent")
    graph.add_edge("knowledge_rag_agent", "reasoning_llm_agent")
    graph.add_edge("reasoning_llm_agent", END)

    return graph


def compile_cardio_graph():
    """Build and compile the graph, returning a runnable."""
    graph = build_cardio_graph()
    return graph.compile()


async def run_pipeline(
    wav_bytes: bytes,
    filename: str = "recording.wav",
    user_prompt: str = "Analyse this heart sound recording.",
) -> dict[str, Any]:
    """Run the full 3-agent pipeline on a .wav recording.

    Parameters
    ----------
    wav_bytes:
        Raw WAV file content.
    filename:
        Original filename for display.
    user_prompt:
        Optional clinical question from the user.

    Returns
    -------
    dict
        The final :class:`AgentState` after all three agents have run.
    """
    app = compile_cardio_graph()
    initial_state: AgentState = {
        "wav_bytes": wav_bytes,
        "filename": filename,
        "user_prompt": user_prompt,
        "errors": [],
    }
    # LangGraph's invoke returns the final state
    result = app.invoke(initial_state)
    return dict(result)

