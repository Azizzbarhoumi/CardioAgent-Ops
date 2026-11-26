"""AudioAgent node — re-exports from agents.audio_agent.agent."""

from agents.audio_agent.agent import (
    audio_agent_node,
    audio_agent_sync,
)

__all__ = ["audio_agent_node", "audio_agent_sync"]
