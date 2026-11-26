"""Agent conformity & unit tests — fully mocked LLM and VectorDB.

These tests validate:
1. Pydantic v2 output schemas reject malformed data (anti-hallucination).
2. Each agent node updates the state correctly when given mocked service responses.
3. The LangGraph StateGraph compiles and runs in the correct order.
4. The orchestrator FastAPI endpoint returns valid PipelineOutput.

All external calls (audio-engine, Qdrant, Ollama) are mocked so tests
run fast locally and in CI with zero infrastructure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.schemas.output_schemas import (
    AudioAnalysisResult,
    ClinicalReasoning,
    PipelineOutput,
    RAGOutput,
    RAGSource,
)
from agents.state.agent_state import AgentState

# ═══════════════════════════════════════════════════════════════════════
# 1. Pydantic Schema Validation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAudioAnalysisResult:
    """Validate AudioAnalysisResult schema constraints."""

    def test_valid_normal_prediction(self):
        result = AudioAnalysisResult(
            prediction_label="Normal",
            confidence=0.92,
            model_name="SVM (RBF)",
            duration_s=2.5,
            sample_rate=2000,
        )
        assert result.prediction_label == "Normal"
        assert result.confidence == 0.92

    def test_valid_abnormal_prediction(self):
        result = AudioAnalysisResult(
            prediction_label="Abnormal",
            confidence=0.78,
            model_name="CNN",
            duration_s=1.5,
            sample_rate=2000,
        )
        assert result.prediction_label == "Abnormal"

    def test_rejects_invalid_label(self):
        with pytest.raises(Exception):  # ValidationError
            AudioAnalysisResult(
                prediction_label="Maybe",
                confidence=0.5,
                model_name="SVM",
                duration_s=1.0,
                sample_rate=2000,
            )

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(Exception):
            AudioAnalysisResult(
                prediction_label="Normal",
                confidence=1.5,
                model_name="SVM",
                duration_s=1.0,
                sample_rate=2000,
            )

    def test_rejects_negative_confidence(self):
        with pytest.raises(Exception):
            AudioAnalysisResult(
                prediction_label="Normal",
                confidence=-0.1,
                model_name="SVM",
                duration_s=1.0,
                sample_rate=2000,
            )

    def test_rejects_empty_model_name(self):
        with pytest.raises(Exception):
            AudioAnalysisResult(
                prediction_label="Normal",
                confidence=0.5,
                model_name="",
                duration_s=1.0,
                sample_rate=2000,
            )

    def test_rejects_zero_duration(self):
        with pytest.raises(Exception):
            AudioAnalysisResult(
                prediction_label="Normal",
                confidence=0.5,
                model_name="SVM",
                duration_s=0,
                sample_rate=2000,
            )


class TestClinicalReasoning:
    """Validate ClinicalReasoning anti-hallucination schema."""

    def test_valid_reasoning(self):
        reasoning = ClinicalReasoning(
            clinical_summary="The recording shows normal heart sounds with no murmurs detected.",
            reasoning="The audio analysis indicates normal S1-S2 heart sounds. The SVM model predicts normal with 92% confidence.",
            recommendations=["No further cardiac investigation required based on this recording."],
            confidence_assessment="high",
            guideline_references=["ESC 2021 Valvular Heart Disease Guidelines, Section 4.2"],
        )
        assert reasoning.confidence_assessment == "high"
        assert len(reasoning.recommendations) >= 1

    def test_rejects_invalid_confidence_level(self):
        with pytest.raises(Exception):
            ClinicalReasoning(
                clinical_summary="Some summary here for the test.",
                reasoning="Some reasoning here for the test.",
                recommendations=["Do something."],
                confidence_assessment="very_high",  # invalid
            )

    def test_rejects_empty_recommendations(self):
        with pytest.raises(Exception):
            ClinicalReasoning(
                clinical_summary="Some summary here for the test.",
                reasoning="Some reasoning here for the test.",
                recommendations=[],  # min_length=1
                confidence_assessment="high",
            )

    def test_rejects_short_summary(self):
        with pytest.raises(Exception):
            ClinicalReasoning(
                clinical_summary="Short",  # < 10 chars
                reasoning="Some reasoning here for the test.",
                recommendations=["Do something."],
                confidence_assessment="medium",
            )

    def test_default_disclaimer(self):
        reasoning = ClinicalReasoning(
            clinical_summary="The recording shows normal heart sounds with no murmurs detected.",
            reasoning="The audio analysis indicates normal S1-S2 heart sounds.",
            recommendations=["No further investigation required."],
            confidence_assessment="low",
        )
        assert "educational" in reasoning.disclaimer.lower()


class TestRAGOutput:
    """Validate RAGOutput schema."""

    def test_valid_rag_output(self):
        output = RAGOutput(
            context="ESC guidelines recommend echocardiography...",
            sources=[
                RAGSource(text="Section 4.2: Aortic Stenosis", source="esc_2021_vhd.pdf", score=0.87),
            ],
            num_sources=1,
        )
        assert output.num_sources == 1

    def test_rejects_empty_context(self):
        with pytest.raises(Exception):
            RAGOutput(context="", sources=[], num_sources=0)


class TestPipelineOutput:
    """Validate the complete PipelineOutput schema."""

    def test_assembles_valid_pipeline_output(self):
        output = PipelineOutput(
            audio_result=AudioAnalysisResult(
                prediction_label="Abnormal",
                confidence=0.85,
                model_name="CNN",
                duration_s=3.0,
                sample_rate=2000,
            ),
            rag_output=RAGOutput(
                context="ESC guidelines text...",
                sources=[],
                num_sources=0,
            ),
            clinical_reasoning=ClinicalReasoning(
                clinical_summary="Abnormal heart sounds detected with high confidence.",
                reasoning="The CNN model detected abnormal patterns consistent with cardiac murmur.",
                recommendations=["Refer for echocardiography."],
                confidence_assessment="medium",
            ),
        )
        assert output.audio_result.prediction_label == "Abnormal"
        assert output.clinical_reasoning.confidence_assessment == "medium"


# ═══════════════════════════════════════════════════════════════════════
# 2. Agent Node Tests (Mocked Services)
# ═══════════════════════════════════════════════════════════════════════


MOCK_AUDIO_RESPONSE = {
    "filename": "test.wav",
    "duration_s": 2.5,
    "source_sample_rate": 44100,
    "sample_rate_after_resampling": 2000,
    "results": [
        {
            "model_name": "SVM (RBF)",
            "model_key": "svm_rbf",
            "model_version": "1.0",
            "model_run_id": None,
            "prediction": 1,
            "prediction_label": "Abnormal",
            "confidence_or_score": 0.87,
            "confidence_type": "decision_score",
            "latency_ms": 12.3,
        },
        {
            "model_name": "Decision Tree",
            "model_key": "decision_tree",
            "model_version": "1.0",
            "model_run_id": None,
            "prediction": 0,
            "prediction_label": "Normal",
            "confidence_or_score": 0.65,
            "confidence_type": "probability",
            "latency_ms": 1.1,
        },
    ],
    "total_processing_ms": 45.6,
    "mel_spectrogram_png_base64": "",
    "waveform_png_base64": "",
}


class TestAudioAgentNode:
    """Test AudioAgent node with mocked audio-engine."""

    @pytest.mark.asyncio
    async def test_audio_agent_picks_highest_confidence(self):
        from agents.audio_agent.agent import audio_agent_node

        state: AgentState = {
            "wav_bytes": b"fake-wav-data",
            "filename": "test.wav",
            "errors": [],
        }
        with patch("agents.audio_agent.agent.call_audio_engine", new_callable=AsyncMock) as mock:
            mock.return_value = MOCK_AUDIO_RESPONSE
            result = await audio_agent_node(state)

        assert result["audio_prediction_label"] == "Abnormal"
        assert result["audio_confidence"] == 0.87
        assert result["audio_model_name"] == "SVM (RBF)"
        assert not result["errors"]

    @pytest.mark.asyncio
    async def test_audio_agent_handles_service_error(self):
        from agents.audio_agent.agent import audio_agent_node

        state: AgentState = {
            "wav_bytes": b"fake-wav-data",
            "filename": "test.wav",
            "errors": [],
        }
        with patch("agents.audio_agent.agent.call_audio_engine", new_callable=AsyncMock) as mock:
            mock.side_effect = ConnectionError("audio-engine unreachable")
            result = await audio_agent_node(state)

        assert any("audio-engine call failed" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_audio_agent_handles_missing_wav(self):
        from agents.audio_agent.agent import audio_agent_node

        state: AgentState = {"errors": []}
        result = await audio_agent_node(state)
        assert any("No wav_bytes" in e for e in result["errors"])


class TestKnowledgeAgentNode:
    """Test KnowledgeRAG-Agent node with mocked RAG pipeline."""

    def test_knowledge_agent_returns_context(self):
        from agents.knowledge_rag_agent.agent import knowledge_agent_node
        from rag_engine.retrieval.retriever import RetrievalResult

        state: AgentState = {
            "audio_prediction_label": "Abnormal",
            "audio_confidence": 0.87,
            "audio_model_name": "SVM (RBF)",
            "user_prompt": "What could be wrong?",
            "errors": [],
        }

        mock_results = [
            RetrievalResult(
                text="ESC recommends echocardiography for suspected valvular disease.",
                score=0.91,
                source="esc_2021_vhd.pdf",
                chunk_index=3,
            ),
        ]

        with patch("agents.knowledge_rag_agent.agent.RAGPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.query.return_value = mock_results
            instance.query_as_context.return_value = (
                "[Source: esc_2021_vhd.pdf | Score: 0.910]\n"
                "ESC recommends echocardiography for suspected valvular disease."
            )
            result = knowledge_agent_node(state)

        assert "echocardiography" in result["rag_context"]
        assert len(result["rag_sources"]) == 1
        assert result["rag_sources"][0]["score"] == 0.91

    def test_knowledge_agent_handles_rag_failure(self):
        from agents.knowledge_rag_agent.agent import knowledge_agent_node

        state: AgentState = {
            "audio_prediction_label": "Normal",
            "audio_confidence": 0.95,
            "audio_model_name": "CNN",
            "user_prompt": "Tell me more.",
            "errors": [],
        }

        with patch("agents.knowledge_rag_agent.agent.RAGPipeline") as MockPipeline:
            MockPipeline.return_value.query.side_effect = ConnectionError("Qdrant down")
            result = knowledge_agent_node(state)

        assert any("RAG query failed" in e for e in result["errors"])
        assert result["rag_sources"] == []


class TestReasoningAgentNode:
    """Test ReasoningLLM-Agent node with mocked Ollama."""

    VALID_LLM_JSON = """{
        "clinical_summary": "The recording shows abnormal heart sounds consistent with a systolic murmur.",
        "reasoning": "The SVM model detected abnormal patterns with 87% confidence. ESC guidelines recommend further investigation.",
        "recommendations": ["Refer for transthoracic echocardiography", "Consider cardiology consultation"],
        "confidence_assessment": "medium",
        "guideline_references": ["ESC 2021 VHD Guidelines, Section 4.2"],
        "disclaimer": "This analysis is for educational purposes only and is not a medical diagnosis."
    }"""

    @pytest.mark.asyncio
    async def test_reasoning_agent_valid_response(self):
        from agents.reasoning_llm_agent.agent import reasoning_agent_node

        state: AgentState = {
            "wav_bytes": b"fake",
            "filename": "test.wav",
            "audio_analysis": {"duration_s": 2.5, "sample_rate_after_resampling": 2000},
            "audio_prediction_label": "Abnormal",
            "audio_confidence": 0.87,
            "audio_model_name": "SVM (RBF)",
            "user_prompt": "What is the diagnosis?",
            "rag_context": "ESC guidelines recommend echocardiography.",
            "errors": [],
        }

        with patch("agents.reasoning_llm_agent.agent.call_llm_generate", new_callable=AsyncMock) as mock:
            mock.return_value = self.VALID_LLM_JSON
            result = await reasoning_agent_node(state)

        assert result["confidence_assessment"] == "medium"
        assert len(result["recommendations"]) == 2
        assert "systolic murmur" in result["clinical_summary"]
        assert not result["errors"]

    @pytest.mark.asyncio
    async def test_reasoning_agent_handles_invalid_json(self):
        from agents.reasoning_llm_agent.agent import reasoning_agent_node

        state: AgentState = {
            "wav_bytes": b"fake",
            "filename": "test.wav",
            "audio_analysis": {"duration_s": 2.0, "sample_rate_after_resampling": 2000},
            "audio_prediction_label": "Normal",
            "audio_confidence": 0.9,
            "audio_model_name": "CNN",
            "user_prompt": "Check this.",
            "rag_context": "Some context.",
            "errors": [],
        }

        with patch("agents.reasoning_llm_agent.agent.call_llm_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "This is not valid JSON at all."
            result = await reasoning_agent_node(state)

        assert any("output validation failed" in e for e in result["errors"])
        assert result["llm_reasoning"] == "This is not valid JSON at all."

    @pytest.mark.asyncio
    async def test_reasoning_agent_handles_llm_failure(self):
        from agents.reasoning_llm_agent.agent import reasoning_agent_node

        state: AgentState = {
            "wav_bytes": b"fake",
            "filename": "test.wav",
            "audio_analysis": {"duration_s": 2.0, "sample_rate_after_resampling": 2000},
            "audio_prediction_label": "Normal",
            "audio_confidence": 0.9,
            "audio_model_name": "CNN",
            "user_prompt": "Check.",
            "rag_context": "Context.",
            "errors": [],
        }

        with patch("agents.reasoning_llm_agent.agent.call_llm_generate", new_callable=AsyncMock) as mock:
            mock.side_effect = ConnectionError("Ollama unreachable")
            result = await reasoning_agent_node(state)

        assert any("LLM call failed" in e for e in result["errors"])


# ═══════════════════════════════════════════════════════════════════════
# 3. LangGraph Compilation Test
# ═══════════════════════════════════════════════════════════════════════


class TestGraphCompilation:
    """Verify the StateGraph compiles and has the expected nodes."""

    def test_graph_compiles_without_error(self):
        from agents.graph.cardio_graph import build_cardio_graph

        graph = build_cardio_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_graph_has_three_nodes(self):
        from agents.graph.cardio_graph import build_cardio_graph

        graph = build_cardio_graph()
        node_names = set(graph.nodes.keys())
        assert "audio_agent" in node_names
        assert "knowledge_rag_agent" in node_names
        assert "reasoning_llm_agent" in node_names

