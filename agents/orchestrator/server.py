"""FastAPI entrypoint for the agent-orchestrator microservice (Service 4).

Exposes a ``/run_pipeline`` endpoint that accepts a .wav file and an
optional user prompt, runs the full 3-agent LangGraph pipeline, and
returns the validated :class:`PipelineOutput`.
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from agents.graph.cardio_graph import run_pipeline
from agents.schemas.output_schemas import (
    AudioAnalysisResult,
    ClinicalReasoning,
    PipelineOutput,
    RAGOutput,
    RAGSource,
)

app = FastAPI(
    title="CardioAgent-Ops Agent Orchestrator",
    description="Service 4 — LangGraph multi-agent pipeline.",
    version="0.1.0",
)


@app.post("/run_pipeline", response_model=PipelineOutput)
async def handle_run_pipeline(
    file: UploadFile = File(...),
    user_prompt: str = Form(default="Analyse this heart sound recording."),
) -> PipelineOutput:
    """Run the full AudioAgent → KnowledgeRAG → ReasoningLLM pipeline."""
    filename = file.filename or "recording.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=415, detail="Upload a .wav audio file.")

    wav_bytes = await file.read()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = await run_pipeline(wav_bytes, filename, user_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    # Assemble validated output
    try:
        analysis = result.get("audio_analysis", {})
        audio_result = AudioAnalysisResult(
            prediction_label=result.get("audio_prediction_label", "Unknown"),
            confidence=result.get("audio_confidence", 0.0),
            model_name=result.get("audio_model_name", "Unknown"),
            duration_s=analysis.get("duration_s", 0),
            sample_rate=analysis.get("sample_rate_after_resampling", 2000),
        )

        rag_sources_raw = result.get("rag_sources", [])
        rag_sources = [RAGSource(**s) for s in rag_sources_raw] if rag_sources_raw else []
        rag_output = RAGOutput(
            context=result.get("rag_context", ""),
            sources=rag_sources,
            num_sources=len(rag_sources),
        )

        clinical = ClinicalReasoning(
            clinical_summary=result.get("clinical_summary", ""),
            reasoning=result.get("llm_reasoning", ""),
            recommendations=result.get("recommendations", []),
            confidence_assessment=result.get("confidence_assessment", "low"),
            guideline_references=[],
        )

        return PipelineOutput(
            audio_result=audio_result,
            rag_output=rag_output,
            clinical_reasoning=clinical,
            errors=result.get("errors", []),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Output validation error: {exc}",
        ) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "agent-orchestrator"}

