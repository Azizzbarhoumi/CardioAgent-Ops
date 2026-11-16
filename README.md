# CardioAgent-Ops

Containerized micro-services platform (Docker), MLOps, LLM fine-tuning, RAG and multi-agent AI for cardio-acoustic analysis.
> **Current status: research notebooks plus a local inference demo.** The project includes an API
> and Streamlit interface for independent SVM, decision-tree, and CNN predictions. Other planned
> platform services remain outside this demo.

## Current project overview

| Area | What is present | Status |
|---|---|---|
| PCG data | 3,785 source recordings from PASCAL and PhysioNet | Inventory and labels can be regenerated with `python scripts/data/build_recordings_manifest.py` |
| SVM | 22 cached features per recording; linear and RBF kernels | Re-runs; 3-fold mean RBF accuracy 0.849 and abnormal-class recall 0.845 |
| Decision tree | 22 cached PCG features | Three-fold stratified baseline; mean accuracy 0.796, abnormal recall 0.855 |
| CNN | `TP7-CNN.ipynb` and exported TensorFlow checkpoint | 40-band Mel image input; mean fold accuracy about 0.731 and abnormal recall 1.000 |
| Inference demo | FastAPI, shared audio preprocessing, Streamlit UI | Four independent outputs: linear SVM, RBF SVM, Decision Tree, and CNN |

The current PCG class distribution is imbalanced: 1,016 normal and 2,769 abnormal recordings.
See [`Data/README.md`](Data/README.md) for the source layout, label mapping, and safe data-handling
notes. The older `01_Tp1_and_Tp2.ipynb` does not match the current folder layout; use `tp1.ipynb`
for the current PASCAL/PhysioNet merge.

## Goal (from the project brief)

Design, evaluate and deploy an end-to-end medical decision platform that combines:

- audio signal processing and benchmarking of 5 ML/DL algorithms, tracked in MLflow, best model exported to ONNX;
- a QLoRA fine-tuned open-source LLM (Qwen-2.5-7B-Instruct or Llama-3.1-8B-Instruct) served locally in quantized form;
- a RAG pipeline over ESC clinical guidelines in a vector database;
- a 3-agent LangGraph network with Pydantic guardrails;
- automated tests (Pytest, coverage) and a GitHub Actions CI/CD pipeline.

## The 6 micro-services

| # | Service | Role | Code lives in |
|---|---------|------|---------------|
| 1 | `llm-server` | Ollama/vLLM serving the quantized fine-tuned LLM | `llm_finetuning/`, `docker/llm-server/` |
| 2 | `vector-db` | Qdrant/ChromaDB for ESC guideline embeddings | `rag_engine/`, `docker/vector-db/` |
| 3 | `audio-engine` | Preprocessing, benchmark, ONNX inference | `audio_engine/`, `docker/audio-engine/` |
| 4 | `agent-orchestrator` | LangGraph agents + Pydantic guardrails | `agents/`, `docker/agent-orchestrator/` |
| 5 | `api-gateway` | FastAPI entry point, `/analyze_pcg` | `api/`, `docker/api-gateway/` |
| 6 | `mlflow-server` | Tracking and Model Registry | `mlops/`, `docker/mlflow-server/` |

## Repository layout

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for the purpose of every directory.
See [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md) for the implementation order and
[`TECHNOLOGY_DECISIONS.md`](TECHNOLOGY_DECISIONS.md) for what is specified versus proposed.

## Local inference demo

### Run the API, UI, and MLflow with Docker Compose

The `audio-engine` service runs the FastAPI inference API; the `streamlit` service provides the UI.
Both use the exported models under `audio_engine/artifacts/models/`. The MLflow service stores its
database and artifacts in a persistent Docker volume and proxies artifact uploads from the app.

```bash
docker compose up --build
```

The UI is available at `http://localhost:8501`, the API at `http://localhost:8000` (health:
`http://localhost:8000/health`), and MLflow at `http://localhost:5000`. Inside Compose, the API
connects to MLflow at `http://mlflow-server:5000`; the browser-facing run link uses
`MLFLOW_PUBLIC_URL` (default `http://localhost:5000`). Use `docker compose down` to stop the
services. The named `mlflow_data` volume persists until explicitly removed; do not use
`docker compose down -v` if you want to retain runs.

Install the API and test dependencies in one environment, then install the UI dependencies:

```bash
python -m pip install -r requirements-api.txt -r requirements-test.txt
python -m pip install -r requirements-frontend.txt
```

Start MLflow in a terminal (MLflow 3.16.1):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Export the latest linear and RBF SVM Logged Models, and create the local Decision Tree export
from the existing feature files. The existing TensorFlow CNN checkpoint is also copied into the
configured model directory:

```bash
python scripts/export_inference_models.py
```

Start the API and UI in separate terminals:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
streamlit run frontend/app.py
```

Set `MLFLOW_TRACKING_URI`, `API_URL`, model paths, and upload/duration limits through environment
variables. `.env.example` lists the supported names and defaults. The API logs each uploaded
recording in a separate `inference_logs` run; if MLflow is unavailable, inference still returns
with a warning.

Run the automated checks with coverage:

```bash
pytest --cov=api --cov=audio_engine --cov-report=term-missing
```

These predictions are for an educational project only and are not intended for medical diagnosis.
