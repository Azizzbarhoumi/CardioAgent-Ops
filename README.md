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
| Inference demo | FastAPI, shared audio preprocessing, React UI (legacy Streamlit retained) | Four independent outputs: linear SVM, RBF SVM, Decision Tree, and CNN |

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

### React web app (current replacement UI)

The React interface lives in `frontend/`; the original Streamlit interface remains available at
`frontend/app.py`. From the project root, create and activate a Python environment, then install
the API and test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt -r requirements-test.txt
```

For MLflow history, artifact logging, and run links, start MLflow in one terminal (from the project
root):

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Start the API in another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

In a third terminal, start the React app:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Set `frontend/.env` to `VITE_API_URL=http://127.0.0.1:8000` if
the API uses another address. MLflow defaults to `http://localhost:5000`; configure
`MLFLOW_TRACKING_URI`, `MLFLOW_PUBLIC_URL`, model paths, and allowed browser origins with
`CORS_ALLOWED_ORIGINS` in `.env` as needed. If MLflow is offline, predictions still return with a
warning, but history and run links are unavailable.

Run the backend/golden checks from the project root:

```powershell
pytest
python scripts/verify_predictions.py
```

Run the React component tests and production build from `frontend/`:

```powershell
npm test
npm run build
```

The Playwright browser check (run from `frontend/`) starts an isolated API and MLflow store,
uploads all golden recordings, compares displayed predictions, checks run metrics and history,
and captures Analyze, Models, and History screenshots:

```powershell
npm run test:e2e
```

It uses the installed Microsoft Edge browser channel and saves screenshots under
`frontend/e2e/screenshots/`. The isolated E2E server uses `.venv\Scripts\python.exe` by default;
set the `PYTHON` environment variable when the project dependencies are installed in another
Python environment.

## Docker Compose (React frontend)

Install and start Docker Desktop, then open PowerShell at the project root. The existing
`audio_engine/artifacts/models/` directory must contain the exported models.

```powershell
cd "C:\Users\azizb\Downloads\CardioAgent-Ops"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up --build
```

Compose starts MLflow, the FastAPI backend, and the React frontend. Open the React app at
`http://localhost:5173`, the API health endpoint at `http://localhost:8000/health`, and MLflow at
`http://localhost:5000`. The app-facing `VITE_API_URL` and `MLFLOW_PUBLIC_URL` default to those
localhost addresses. If you change them, use addresses reachable by the browser; Compose-only
hostnames such as `audio-engine` are not browser URLs.

Press `Ctrl+C` to stop the foreground Compose run, then use `docker compose down`. MLflow data is
kept in the `mlflow_data` volume; do not add `-v` if you want to retain it.

The original Streamlit UI is retained as an optional legacy service. To start it alongside the
React app, use:

```powershell
docker compose --profile legacy up --build
```

The Streamlit app is then available at `http://localhost:8501`.

These predictions are for an educational project only and are not intended for medical diagnosis.
