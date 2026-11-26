"""HTTP clients for inter-service communication.

Each agent calls a different microservice:
- AudioAgent → audio-engine (Service 3)
- KnowledgeRAG-Agent → vector-db / rag_engine (Service 2)
- ReasoningLLM-Agent → llm-server / Ollama (Service 1)
"""

from __future__ import annotations

import httpx

from agents.configs.agent_config import agent_config


async def call_audio_engine(wav_bytes: bytes, filename: str = "recording.wav") -> dict:
    """POST a .wav file to the audio-engine /analyze_pcg endpoint."""
    async with httpx.AsyncClient(timeout=agent_config.audio_engine_timeout) as client:
        response = await client.post(
            f"{agent_config.audio_engine_url}/analyze_pcg",
            files={"file": (filename, wav_bytes, "audio/wav")},
        )
        response.raise_for_status()
        return response.json()


async def call_llm_generate(prompt: str, system: str = "") -> str:
    """POST a prompt to the Ollama /api/generate endpoint."""
    payload = {
        "model": agent_config.llm_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": agent_config.llm_temperature,
            "num_predict": agent_config.llm_max_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=agent_config.llm_timeout) as client:
        response = await client.post(
            f"{agent_config.llm_server_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json().get("response", "")

