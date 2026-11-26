"""Service entrypoint for Service 4 (agent-orchestrator) with MLflow tracing."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.communication.streaming import stream_cardio_workflow
from agents.configs.settings import AgentConfig
from agents.graph.workflow import run_cardio_workflow


class OrchestratorRequest(BaseModel):
    filename: str = "recording.wav"
    patient_context: str = ""
    mock_audio: bool = Field(default_factory=lambda: os.getenv("MOCK_AUDIO", "false").lower() in {"1", "true", "yes"})


def log_agent_trace_to_mlflow(result: dict[str, Any], tracking_uri: str) -> str | None:
    """Best-effort logging of agent execution trace to MLflow experiment `agent_traces`."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("agent_traces")
        with mlflow.start_run(run_name="langgraph_cardio_trace") as run:
            audio = result.get("audio_result", {})
            report = result.get("reasoning_report", {})
            mlflow.log_params(
                {
                    "audio_source": audio.get("source", "unknown"),
                    "prediction": audio.get("prediction", "unknown"),
                    "suspected_timing": audio.get("suspected_timing", "unknown"),
                    "risk_level": report.get("risk_level", "unknown"),
                }
            )
            if "confidence" in audio:
                mlflow.log_metric("audio_confidence", float(audio["confidence"]))
            mlflow.log_dict(result, "agent_workflow_result.json")
            return run.info.run_id
    except Exception:
        return None


app = FastAPI(title="CardioAgent Orchestrator", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = AgentConfig()
    return {"status": "ok", "mock_audio": cfg.mock_audio, "llm_model": cfg.llm_model}


@app.post("/orchestrate")
def orchestrate(request: OrchestratorRequest) -> dict[str, Any]:
    cfg = AgentConfig()
    result = run_cardio_workflow(
        {
            "filename": request.filename,
            "patient_context": request.patient_context,
            "mock_audio": request.mock_audio,
        },
        config=cfg,
    )
    run_id = log_agent_trace_to_mlflow(result, cfg.mlflow_tracking_uri)
    result["mlflow_run_id"] = run_id
    return result


@app.post("/orchestrate/stream")
def orchestrate_stream(request: OrchestratorRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_cardio_workflow(
            {
                "filename": request.filename,
                "patient_context": request.patient_context,
                "mock_audio": request.mock_audio,
            }
        ),
        media_type="text/event-stream",
    )
