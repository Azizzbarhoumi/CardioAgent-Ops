"""Server-Sent Events (SSE) formatting helpers for agent workflow streaming."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from agents.graph.workflow import build_cardio_graph
from agents.state.graph_state import CardioAgentState


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    """Format a dictionary payload as an SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_cardio_workflow(initial_state: CardioAgentState) -> Iterator[str]:
    """Yield SSE frames as each LangGraph node completes."""
    graph = build_cardio_graph()
    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            yield format_sse_event(node_name, node_output)
    yield format_sse_event("done", {"status": "completed"})
