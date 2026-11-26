"""Shared helper utilities for agent nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def append_trace(state: dict[str, Any], node_name: str, status: str, detail: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Append an immutable step trace entry to the workflow trace list."""
    trace = list(state.get("trace") or [])
    trace.append(
        {
            "node": node_name,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detail": detail or {},
        }
    )
    return trace
