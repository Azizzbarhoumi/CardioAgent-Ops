"""LangGraph 3-agent StateGraph: AudioAgent -> KnowledgeRAGAgent -> ReasoningLLMAgent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.audio_agent.agent import AudioAgent
from agents.configs.settings import AgentConfig
from agents.knowledge_rag_agent.agent import KnowledgeRAGAgent
from agents.reasoning_llm_agent.agent import ReasoningLLMAgent
from agents.state.graph_state import CardioAgentState
from rag_engine.pipeline.rag_pipeline import CardioRAGPipeline


def build_cardio_graph(
    *,
    config: AgentConfig | None = None,
    rag_pipeline: CardioRAGPipeline | None = None,
):
    """Build and compile the 3-agent LangGraph workflow."""
    cfg = config or AgentConfig()
    audio_agent = AudioAgent(config=cfg)
    rag_agent = KnowledgeRAGAgent(rag_pipeline=rag_pipeline, config=cfg)
    reasoning_agent = ReasoningLLMAgent(config=cfg)

    graph = StateGraph(CardioAgentState)
    graph.add_node("audio_agent", audio_agent.run)
    graph.add_node("knowledge_rag_agent", rag_agent.run)
    graph.add_node("reasoning_llm_agent", reasoning_agent.run)

    graph.add_edge(START, "audio_agent")
    graph.add_edge("audio_agent", "knowledge_rag_agent")
    graph.add_edge("knowledge_rag_agent", "reasoning_llm_agent")
    graph.add_edge("reasoning_llm_agent", END)
    return graph.compile()


def run_cardio_workflow(
    initial_state: CardioAgentState | None = None,
    *,
    config: AgentConfig | None = None,
    rag_pipeline: CardioRAGPipeline | None = None,
) -> dict[str, Any]:
    """Execute the compiled 3-agent LangGraph workflow synchronously."""
    compiled = build_cardio_graph(config=config, rag_pipeline=rag_pipeline)
    return compiled.invoke(initial_state or {})
