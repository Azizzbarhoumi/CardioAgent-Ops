# Project Structure

Every directory below exists in the repository. **Origin** tells you where it comes from:
- **PDF**: required by or directly derived from the project brief.
- **Proposed**: an organizational choice of this scaffold, not a requirement of the brief.

Empty leaf directories contain a `.gitkeep` file so Git keeps them.

## Full tree

```
cardioagent-ops/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
├── agents/
│   ├── audio_agent/
│   ├── communication/
│   ├── configs/
│   ├── graph/
│   ├── knowledge_rag_agent/
│   ├── orchestrator/
│   ├── prompts/
│   ├── reasoning_llm_agent/
│   ├── schemas/
│   ├── shared/
│   ├── state/
│   └── tools/
├── api/
│   ├── clients/
│   ├── core/
│   ├── guardrails/
│   ├── routers/
│   ├── schemas/
│   ├── services/
│   └── streaming/
├── audio_engine/
│   ├── artifacts/
│   │   └── models/
│   ├── configs/
│   ├── evaluation/
│   ├── export/
│   ├── features/
│   ├── inference/
│   ├── ingestion/
│   ├── preprocessing/
│   └── training/
│       ├── cnn_custom/
│       ├── decision_tree/
│       ├── resnet_transfer/
│       ├── svm/
│       └── transformer_cnn/
├── data/
│   ├── external/
│   │   ├── esc_guidelines/
│   │   └── medical_corpus/
│   ├── interim/
│   ├── labels/
│   ├── processed/
│   │   ├── audio/
│   │   ├── features/
│   │   │   ├── mel_spectrograms/
│   │   │   ├── mfcc/
│   │   │   └── zcr/
│   │   └── splits/
│   ├── raw/
│   │   ├── pascal/
│   │   └── physionet/
│   └── validation/
├── docker/
│   ├── agent-orchestrator/
│   ├── api-gateway/
│   ├── audio-engine/
│   ├── env/
│   ├── llm-server/
│   ├── mlflow-server/
│   ├── overrides/
│   └── vector-db/
├── docs/
│   ├── api/
│   ├── architecture/
│   │   └── diagrams/
│   ├── guidelines/
│   ├── mlops/
│   ├── report/
│   │   ├── benchmarks/
│   │   ├── coverage/
│   │   ├── figures/
│   │   ├── mlflow_screenshots/
│   │   ├── rag_evaluation/
│   │   └── resource_usage/
│   └── specifications/
├── llm_finetuning/
│   ├── artifacts/
│   ├── configs/
│   ├── dataset/
│   │   ├── processed/
│   │   └── raw/
│   ├── evaluation/
│   ├── export/
│   │   ├── gguf/
│   │   └── ollama/
│   ├── inference/
│   └── training/
├── mlops/
│   ├── continuous_training/
│   ├── data_versioning/
│   ├── deployment/
│   ├── mlflow_server/
│   ├── model_registry/
│   ├── monitoring/
│   ├── pipelines/
│   └── tracking/
├── notebooks/
├── rag_engine/
│   ├── chunking/
│   ├── configs/
│   ├── embeddings/
│   ├── evaluation/
│   ├── ingestion/
│   ├── parsing/
│   ├── pipeline/
│   ├── retrieval/
│   └── vector_store/
├── scripts/
│   ├── data/
│   ├── deployment/
│   ├── maintenance/
│   ├── setup/
│   └── training/
├── shared/
│   ├── constants/
│   ├── schemas/
│   └── utils/
├── tests/
│   ├── agents/
│   ├── api/
│   ├── e2e/
│   ├── fixtures/
│   ├── integration/
│   ├── model/
│   ├── performance/
│   └── unit/
│       ├── audio/
│       ├── llm/
│       └── rag/
├── .gitignore
├── DEVELOPMENT_ROADMAP.md
├── PROJECT_STRUCTURE.md
├── README.md
└── TECHNOLOGY_DECISIONS.md
```

## Root files

| File | Purpose |
|---|---|
| `README.md` | Project overview and entry point |
| `PROJECT_STRUCTURE.md` | This document |
| `DEVELOPMENT_ROADMAP.md` | Logical implementation order |
| `TECHNOLOGY_DECISIONS.md` | Specified vs proposed vs unresolved technologies |
| `.gitignore` | Keeps secrets, datasets and large model artifacts out of Git |

## `.github/` — CI/CD and repository templates

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `.github/` | PDF | GitHub repository metadata and automation. |
| `.github/workflows/` | PDF | GitHub Actions workflows. The PDF mandates `mlops-ci-cd.yml` (tests, coverage check, API Gateway image build) to be created here later. |
| `.github/ISSUE_TEMPLATE/` | Proposed | Issue templates (bug, task) so the team reports work in a consistent format. |

## `agents/` — Multi-agent orchestration (Service 4)

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `agents/` | PDF | Source of Service 4 `agent-orchestrator`: LangGraph multi-agent graph, prompts, Pydantic schemas, guardrails. |
| `agents/configs/` | Proposed | Agent and orchestrator settings (service URLs, timeouts, model names). |
| `agents/graph/` | PDF | LangGraph StateGraph definition (nodes, edges). |
| `agents/state/` | PDF | Shared state schema passed between agents. |
| `agents/audio_agent/` | PDF | AudioAgent: calls `audio-engine` and produces the audio analysis. |
| `agents/knowledge_rag_agent/` | PDF | KnowledgeRAGAgent: queries `vector-db` for guideline context. |
| `agents/reasoning_llm_agent/` | PDF | ReasoningLLMAgent: calls `llm-server` to produce the medical reasoning. |
| `agents/prompts/` | PDF | Prompt templates for the agents. |
| `agents/schemas/` | PDF | Pydantic v2 schemas validating agent inputs/outputs (anti-hallucination guardrails). |
| `agents/tools/` | Proposed | Tool wrappers used by agents (HTTP clients to the other services). |
| `agents/communication/` | Proposed | Inter-agent messaging and token streaming (SSE) back to the gateway. |
| `agents/orchestrator/` | PDF | Service entrypoint exposing the graph to the API Gateway and sending traces to MLflow. |
| `agents/shared/` | Proposed | Components reused across agents. |

## `api/` — API Gateway (Service 5)

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `api/` | PDF | Source of Service 5 `api-gateway` (FastAPI/Uvicorn): single entry point exposing `/analyze_pcg` (`main.py`). |
| `api/routers/` | PDF | Route definitions, including `/analyze_pcg`. |
| `api/guardrails/` | PDF | Input/output validation and guardrails at the gateway. |
| `api/schemas/` | PDF | Pydantic request/response models. |
| `api/clients/` | Proposed | HTTP clients to `audio-engine` and `agent-orchestrator`. |
| `api/services/` | Proposed | Business logic for forwarding .wav requests between client, audio engine and orchestrator. |
| `api/streaming/` | PDF | Streaming response / Server-Sent Events handling. |
| `api/core/` | Proposed | App configuration, logging, dependency wiring. |

## `audio_engine/` — Audio ML pipeline (Service 3)

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `audio_engine/` | PDF | Source of Service 3 `audio-engine`: audio preprocessing, benchmark of the 5 algorithms, MLflow tracking, ONNX inference. |
| `audio_engine/configs/` | Proposed | Hyperparameters and pipeline settings (sampling rate, duration, feature options). |
| `audio_engine/ingestion/` | Proposed | Loading and merging of PASCAL and PhysioNet audio and labels. |
| `audio_engine/preprocessing/` | PDF | Signal preprocessing: resampling to 2000 Hz, duration normalization, amplitude normalization. |
| `audio_engine/features/` | PDF | Feature extraction: MFCC, ZCR, Mel spectrograms. |
| `audio_engine/training/` | PDF | Training scripts for the benchmarked algorithms, with MLflow logging. |
| `audio_engine/training/decision_tree/` | PDF | Algorithm 1: Decision Tree on MFCC features. |
| `audio_engine/training/svm/` | PDF | Algorithm 2: SVM with RBF kernel. |
| `audio_engine/training/cnn_custom/` | PDF | Algorithm 3: custom 2D CNN on Mel spectrograms. |
| `audio_engine/training/resnet_transfer/` | PDF | Algorithm 4: transfer learning (ResNet18/50 or VGG16 backbone). |
| `audio_engine/training/transformer_cnn/` | PDF | Algorithm 5: Transformer-CNN hybrid (AST / Conformer). |
| `audio_engine/evaluation/` | PDF | Metrics and comparison of models; selection of the model maximizing recall on the pathological class. |
| `audio_engine/export/` | PDF | Export of the selected model to ONNX (`cardio_model.onnx`). |
| `audio_engine/inference/` | PDF | ONNX Runtime inference code used by the service. |
| `audio_engine/artifacts/` | Proposed | Local model artifacts (git-ignored; MLflow is the source of truth). |
| `audio_engine/artifacts/models/` | PDF | Exported models, e.g. `cardio_model.onnx`. |

## `data/` — Datasets (not versioned in Git)

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `data/` | Proposed | All datasets. Content is not versioned in Git (large/sensitive); only the structure is. |
| `data/raw/` | PDF | Original, untouched datasets. |
| `data/raw/pascal/` | PDF | PASCAL Heart Sound Challenge dataset. |
| `data/raw/physionet/` | PDF | PhysioNet/CinC Challenge dataset. |
| `data/interim/` | Proposed | Intermediate outputs of the merge/cleaning stage. |
| `data/processed/` | PDF | Model-ready data after wrangling. |
| `data/processed/audio/` | PDF | Merged, harmonized audio: resampled to 2000 Hz and normalized to fixed duration (2 s or 5 s). |
| `data/processed/features/` | PDF | Extracted features. |
| `data/processed/features/mfcc/` | PDF | MFCC features (1D) for Decision Tree and SVM. |
| `data/processed/features/zcr/` | PDF | Zero-crossing-rate features. |
| `data/processed/features/mel_spectrograms/` | PDF | Mel spectrograms (2D) for CNN-based models. |
| `data/processed/splits/` | Proposed | Train/validation/test split definitions to guarantee reproducibility. |
| `data/labels/` | Proposed | Unified labels and annotations (normal / pathological) after harmonizing both datasets. |
| `data/external/` | Proposed | Third-party non-audio sources. |
| `data/external/esc_guidelines/` | PDF | ESC (European Society of Cardiology) clinical guideline documents to be indexed by the RAG pipeline. |
| `data/external/medical_corpus/` | PDF | Medical corpus used to build the LLM instruction dataset. |
| `data/validation/` | Proposed | Data validation reports and checks (class balance, duration, sampling rate, duplicates). |

## `docker/` — Containerization

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `docker/` | PDF | Docker infrastructure; one Dockerfile per micro-service and the `docker-compose.yml` (to be created). |
| `docker/llm-server/` | PDF | Service 1: Ollama or vLLM serving the quantized fine-tuned LLM. |
| `docker/vector-db/` | PDF | Service 2: Qdrant or ChromaDB. |
| `docker/audio-engine/` | PDF | Service 3: Python image with ONNX Runtime and Librosa. |
| `docker/agent-orchestrator/` | PDF | Service 4: Python image hosting the LangGraph graph and Pydantic guardrails. |
| `docker/api-gateway/` | PDF | Service 5: FastAPI/Uvicorn image. |
| `docker/mlflow-server/` | PDF | Service 6: MLflow image. |
| `docker/env/` | Proposed | Environment templates (no secrets committed). |
| `docker/overrides/` | Proposed | Optional compose overrides (development, CI). |

## `docs/` — Documentation and technical report

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `docs/` | Proposed | All written documentation, kept separate from code. |
| `docs/architecture/` | PDF | System architecture: the 6 micro-services, docker-network, LangGraph agent flow, data flow. Diagrams and design notes go here. |
| `docs/architecture/diagrams/` | Proposed | Source files and exports of architecture diagrams. |
| `docs/specifications/` | PDF | Technical specifications derived from the PDF: requirements, service contracts, acceptance criteria. |
| `docs/mlops/` | PDF | MLOps pipeline documentation: MLflow tracking and registry usage, CI/CD/CT, model promotion rules. |
| `docs/api/` | PDF | API documentation for the `/analyze_pcg` endpoint (request/response schema, SSE streaming behavior). |
| `docs/guidelines/` | Proposed | Development guidelines: branching, code style, commit conventions, how to run each module. |
| `docs/report/` | PDF | Technical report deliverable (PDF/README): benchmark, MLflow tracking, coverage, RAG evaluation, architecture, memory/VRAM usage. |
| `docs/report/figures/` | PDF | Figures and schemas used in the report (including the 6 micro-services diagram). |
| `docs/report/benchmarks/` | PDF | Detailed benchmark tables and plots of the audio ML/DL algorithms. |
| `docs/report/mlflow_screenshots/` | PDF | Screenshots of MLflow experiments, runs and model registry. |
| `docs/report/coverage/` | PDF | pytest-cov coverage reports. |
| `docs/report/rag_evaluation/` | PDF | RAG evaluation results. |
| `docs/report/resource_usage/` | PDF | Memory and VRAM consumption measurements per service. |

## `llm_finetuning/` — LLM fine-tuning (feeds Service 1)

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `llm_finetuning/` | PDF | LLM fine-tuning module (QLoRA 4-bit, Unsloth/PEFT/TRL) for the cardiology-specialized model served by `llm-server`. |
| `llm_finetuning/configs/` | Proposed | Training configuration (base model, QLoRA parameters, quantization). |
| `llm_finetuning/dataset/` | PDF | Medical instruction dataset in JSONL. |
| `llm_finetuning/dataset/raw/` | Proposed | Source texts before conversion to instructions. |
| `llm_finetuning/dataset/processed/` | PDF | Final JSONL instruction dataset. |
| `llm_finetuning/training/` | PDF | QLoRA training scripts with loss tracking in MLflow. |
| `llm_finetuning/evaluation/` | Proposed | Evaluation of the fine-tuned model (output quality, schema compliance). |
| `llm_finetuning/export/` | PDF | Export artifacts for the LLM server. |
| `llm_finetuning/export/gguf/` | PDF | Quantized GGUF model files. |
| `llm_finetuning/export/ollama/` | PDF | Ollama model definition files for loading the adapter / GGUF model. |
| `llm_finetuning/artifacts/` | Proposed | LoRA adapters and checkpoints (git-ignored). |
| `llm_finetuning/inference/` | Proposed | Client code and scripts to query the LLM server and track inference metrics. |

## `mlops/` — MLOps (Service 6 and tracking)

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `mlops/` | PDF | MLOps layer: tracking, registry, continuous training, monitoring. |
| `mlops/mlflow_server/` | PDF | Configuration for Service 6 `mlflow-server` (tracking + Model Registry). |
| `mlops/tracking/` | PDF | Conventions and helpers for logging experiments (metrics, hyperparameters, artifacts). |
| `mlops/model_registry/` | PDF | Model registry usage: stages, promotion rules. |
| `mlops/pipelines/` | Proposed | Orchestrated training pipelines (audio, LLM). |
| `mlops/continuous_training/` | PDF | Continuous Training (CT) definitions. |
| `mlops/monitoring/` | PDF | Inference metrics tracking and resource monitoring (memory/VRAM). |
| `mlops/deployment/` | Proposed | Model promotion and deployment procedures. |
| `mlops/data_versioning/` | Proposed | Data versioning setup (tool to be decided). |

## `notebooks/` — Notebooks

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `notebooks/` | PDF | Jupyter notebooks. The PDF mandates `01_eda_data_wrangling.ipynb` (merge PASCAL + PhysioNet, cleaning, visualization, data mining). |

## `rag_engine/` — RAG pipeline (feeds Service 2)

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `rag_engine/` | PDF | RAG module: indexing ESC clinical guidelines in the vector database for grounded medical reasoning. |
| `rag_engine/configs/` | Proposed | Chunking, embedding model and retrieval settings. |
| `rag_engine/ingestion/` | Proposed | Document loading from `data/external/esc_guidelines`. |
| `rag_engine/parsing/` | Proposed | Parsing and cleaning of guideline documents. |
| `rag_engine/chunking/` | PDF | Chunking strategies. |
| `rag_engine/embeddings/` | PDF | Dense embedding generation (Sentence-Transformers). |
| `rag_engine/vector_store/` | PDF | Qdrant or ChromaDB client and collection setup (Service 2 `vector-db`). |
| `rag_engine/retrieval/` | PDF | Cosine-similarity retrieval logic. |
| `rag_engine/pipeline/` | Proposed | End-to-end RAG pipeline assembly used by the KnowledgeRAG agent. |
| `rag_engine/evaluation/` | PDF | RAG evaluation (retrieval quality, grounding). |

## `scripts/` — Scripts

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `scripts/` | Proposed | Helper scripts. |
| `scripts/setup/` | Proposed | Environment and repository setup. |
| `scripts/data/` | Proposed | Data download and preparation. |
| `scripts/training/` | Proposed | Launchers for audio and LLM training. |
| `scripts/deployment/` | Proposed | Build and deploy helpers. |
| `scripts/maintenance/` | Proposed | Cleanup and housekeeping. |

## `shared/` — Shared code

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `shared/` | Proposed | Code shared by several services, to avoid duplication. |
| `shared/schemas/` | Proposed | Common data schemas used across services. |
| `shared/utils/` | Proposed | Common utilities (logging, config loading, HTTP helpers). |
| `shared/constants/` | Proposed | Project-wide constants (sampling rate, labels, service names). |

## `tests/` — Automated tests

| Directory | Origin | Purpose / what will be developed |
|---|---|---|
| `tests/` | PDF | Pytest suite with coverage measured by pytest-cov. |
| `tests/unit/` | PDF | Unit tests. |
| `tests/unit/audio/` | PDF | Audio processing tests (`test_audio.py`): resampling, normalization, features. |
| `tests/unit/rag/` | Proposed | Chunking/embedding/retrieval unit tests. |
| `tests/unit/llm/` | Proposed | LLM dataset and inference-client unit tests. |
| `tests/integration/` | PDF | Service-to-service integration tests. |
| `tests/api/` | PDF | API Gateway tests with HTTPX (`test_api.py`). |
| `tests/model/` | Proposed | Model tests: ONNX output consistency, minimum recall. |
| `tests/agents/` | PDF | Agent graph and LLM schema-conformance tests (`test_agents.py`). |
| `tests/e2e/` | Proposed | End-to-end tests through the full docker compose stack. |
| `tests/performance/` | Proposed | Latency and resource tests. |
| `tests/fixtures/` | Proposed | Small shared test inputs. |

## Notes

- The brief mandates these deliverable names, which the layout follows: `01_eda_data_wrangling.ipynb`, `audio_engine/`, `llm_finetuning/`, `rag_engine/`, `agents/`, `api/` (`main.py`), `docker/`, `tests/` (`test_audio.py`, `test_api.py`, `test_agents.py`), `.github/workflows/mlops-ci-cd.yml`, `cardio_model.onnx`.
- `docker-compose.yml` is expected inside `docker/` as in the brief. Running it from the repo root would use `docker compose -f docker/docker-compose.yml up --build`; see `TECHNOLOGY_DECISIONS.md`.
- Service 4 `agent-orchestrator` is built from `agents/`, and Service 5 `api-gateway` from `api/`. Each Dockerfile lives in its own `docker/<service>/` folder.
