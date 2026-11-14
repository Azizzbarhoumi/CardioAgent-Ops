# Technology Decisions

Source of truth: the CardioAgent-Ops project brief (PDF). Anything not in the brief is marked **Proposed**.

## 1. Specified by the PDF

| Area | Technology / requirement |
|---|---|
| Containers | Docker, Docker Compose, one Dockerfile per service (6 services), internal `docker-network`, multi-stage Dockerfiles |
| Language | Python 3.10+ |
| Audio / data | Librosa, Pandas, NumPy, SciPy; resample to 2000 Hz; fixed duration (2 s or 5 s); MFCC, ZCR, Mel spectrograms |
| Datasets | PASCAL Heart Sound Challenge, PhysioNet/CinC Challenge |
| ML / DL | Scikit-Learn, PyTorch, Torchvision, ONNX Runtime; export best model to ONNX; select by recall on the pathological class |
| Algorithms (5 and more) | Decision Tree (MFCC), SVM RBF, custom 2D CNN (Mel), transfer learning (ResNet18/50 or VGG16), Transformer-CNN hybrid (AST / Conformer) |
| MLOps | MLflow 2.x (tracking, Model Registry, inference traces) |
| Testing | Pytest, Pytest-Cov, HTTPX; `test_audio.py`, `test_api.py`, `test_agents.py` |
| CI/CD | GitHub Actions, `.github/workflows/mlops-ci-cd.yml` |
| LLM fine-tuning | Unsloth, PEFT, TRL, BitsAndBytes, HuggingFace; QLoRA 4-bit; Qwen-2.5-7B-Instruct or Llama-3.1-8B-Instruct; JSONL dataset; GGUF export |
| LLM serving | Ollama or vLLM (dockerized), INT4/INT8 quantized, no data leakage |
| RAG | Qdrant or ChromaDB, Sentence-Transformers, ESC guidelines, cosine similarity |
| Agents | LangGraph / LangChain, 3 agents, shared state |
| Validation | Pydantic v2 |
| API | FastAPI, Uvicorn, endpoint `/analyze_pcg`, streaming via Server-Sent Events |

## 2. Choices the PDF leaves open (decide and record here)

| Decision | Options in the PDF | Chosen |
|---|---|---|
| Vector database | Qdrant or ChromaDB | _to decide_ |
| LLM server | Ollama or vLLM | _to decide_ |
| Base LLM | Qwen-2.5-7B-Instruct or Llama-3.1-8B-Instruct | _to decide_ |
| Fixed audio duration | 2 s or 5 s | _to decide_ |
| Transfer-learning backbone | ResNet18, ResNet50 or VGG16 | _to decide_ |
| Transformer-CNN model | AST or Conformer | _to decide_ |
| Embedding model | not named (only "Sentence-Transformers") | _to decide_ |
| Location of `docker-compose.yml` | in `docker/` per the brief; the run command is given without a path | _to decide_ |

## 3. Proposed (not in the PDF)

| Item | Why proposed |
|---|---|
| `shared/` package | Avoid duplicated schemas/utilities across services |
| `mlops/data_versioning/` (tool undecided, e.g. DVC) | Datasets are too large for Git |
| `.gitignore` rules for `*.wav`, `*.onnx`, `*.gguf`, weights | Keep large files and secrets out of Git |
| `docker/env/`, `docker/overrides/` | Environment templates and dev/CI compose overrides |
| `tests/model/`, `tests/e2e/`, `tests/performance/`, `tests/fixtures/` | Extra test layers beyond the three mandated test files |
| Issue templates, `scripts/` helpers, `docs/guidelines/` | Team workflow |

## 4. Not in the PDF, deliberately left out

- **Frontend**: the brief defines the client as a `.WAV` + prompt sent to the API (with SSE streaming back). No UI is described, so no `frontend/` folder was created. It can be added later if the team decides to.
- **Authentication/authorization and a relational database**: not mentioned; the gateway only has guardrails and validation.

## 5. Ambiguities in the PDF

- Agent names differ: the diagram shows *AudioAgent → KnowledgeRAGAgent → ReasoningLLMAgent*, while the text says *Analyste → Cardiologue RAG → Auditeur*. The scaffold uses the diagram names (`agents/audio_agent`, `knowledge_rag_agent`, `reasoning_llm_agent`); confirm the mapping with the teacher.
- Service 3 `audio-engine` is described as doing both benchmarking/training and inference. Training code is placed in `audio_engine/training/` and may run outside the container, with only inference packaged.
