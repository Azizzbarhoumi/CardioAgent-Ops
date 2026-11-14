# Development Roadmap

Order follows the 8 milestones of the brief, arranged so that every phase unblocks the next. Deadline: **1 December 2026**.

## Suggested scope tiers

Because of the deadline, work in tiers and finish each before starting the next.

| Tier | Content |
|---|---|
| **A: Core** | Phases 0 to 3, 6, 7 (data, audio models, MLflow, ONNX, API, tests, 3-service Docker, CI) and the report |
| **B: Extended** | RAG (phase 4) and a minimal LangGraph graph (phase 5) |
| **C: Full brief** | Remaining 2 algorithms (ResNet/VGG, Transformer-CNN), QLoRA fine-tuning and GGUF export, all 6 services in compose |

The brief asks for 5 algorithms. Starting with Decision Tree, SVM and CNN is fine, but the missing ones should be documented as future work or added if time allows.

## Phases

### Phase 0: Repository setup
Initialize Git, agree on branching and conventions (`docs/guidelines/`), decide the open choices in `TECHNOLOGY_DECISIONS.md`.

### Phase 1: Data wrangling and mining (milestone 1)
Merge PASCAL + PhysioNet, resample to 2000 Hz, normalize duration, extract MFCC/ZCR/Mel features, balance classes, validate. Output: `notebooks/01_eda_data_wrangling.ipynb`, `data/processed/`.
Folders: `data/`, `notebooks/`, `audio_engine/ingestion|preprocessing|features/`.

### Phase 2: Audio benchmark, MLflow, ONNX (milestone 2)
Stand up `mlflow-server`; train and log the algorithms; select the best recall; export `cardio_model.onnx`.
Folders: `audio_engine/training|evaluation|export/`, `mlops/`.

### Phase 3: Inference and API Gateway
ONNX inference in `audio_engine/inference/`; FastAPI `/analyze_pcg` with Pydantic schemas.
Folders: `audio_engine/inference/`, `api/`, `shared/`.

### Phase 4: RAG and vector database (milestone 4)
Parse, chunk, embed and index ESC guidelines; evaluate retrieval.
Folders: `rag_engine/`, `data/external/esc_guidelines/`.

### Phase 5: LLM fine-tuning (milestone 3) and agents (milestone 5)
These can run in parallel by different people. Build the JSONL dataset, QLoRA training tracked in MLflow, GGUF export; then the 3-agent LangGraph graph with Pydantic schemas.
Folders: `llm_finetuning/`, `agents/`.

### Phase 6: Tests (milestone 6)
Pytest suites for audio, API and agent schemas; coverage with pytest-cov. Write tests alongside phases 1 to 5, not only at the end.
Folders: `tests/`.

### Phase 7: Docker orchestration (milestone 7)
One Dockerfile per service, multi-stage, then the compose file connecting services on `docker-network`.
Folders: `docker/`.

### Phase 8: CI/CD (milestone 8)
GitHub Actions: install, test with coverage, build the API Gateway image.
Folders: `.github/workflows/`.

### Phase 9: Technical report
Benchmark, MLflow screenshots, coverage, RAG evaluation, architecture diagram, memory/VRAM usage.
Folders: `docs/report/`.

## Parallel work split

| Track | Folders |
|---|---|
| Data and audio ML | `data/`, `audio_engine/`, `notebooks/` |
| LLM and RAG | `llm_finetuning/`, `rag_engine/` |
| Agents and API | `agents/`, `api/`, `shared/` |
| Platform and quality | `docker/`, `mlops/`, `tests/`, `.github/` |
