# CardioAgent-Ops: Assessment of the Project Brief

Source: *Mini Projet 2026-2027, Data Mining & Machine Learning* (ENSIT, section 3G Math, Dr. Mehrez Boulares).
Deadline: **1 December 2026**. Assessment date: **15 November 2026** (about 16 days left).

Statuses come from what has been reported so far. Update them as you go.

Legend: ✅ done · 🟡 partly done / to verify · ⬜ to do · ⛔ out of scope (documented as future work)

---

## 1. Summary

| Area | Status | Comment |
|---|---|---|
| Data wrangling | ✅ | Merged dataset ready |
| Audio ML benchmark | 🟡 | 3 of 5 algorithms |
| MLflow tracking | 🟡 | SVM runs logged; check Decision Tree and CNN |
| ONNX export | ⬜ | Required by the brief |
| Inference UI (supervisor request) | ✅ | Streamlit, 3 model groups, MLflow link |
| Docker | 🟡 | MLflow container only |
| Tests + coverage | ⬜ | |
| CI/CD | ⬜ | |
| Technical report | ⬜ | |
| LLM fine-tuning, RAG, agents | ⛔ | Not planned |

Target for this deadline: about 70% of the brief (core MLOps path, no LLM/RAG/agents).

---

## 2. Requirement checklist

### 2.1 Data engineering and mining (milestone 1)

| Requirement | Status | Notes |
|---|---|---|
| Merge PASCAL + PhysioNet | ✅ | |
| Resample to 2000 Hz | ✅ | |
| Normalize duration (2 s or 5 s) | 🟡 | Check what the CNN does with long files (a 35 s test file was accepted) |
| Features: MFCC, ZCR, spectrograms | 🟡 | Current tabular features are duration + SNR + 20 Mel means, not MFCC/ZCR |
| Cleaning, visualization, class balance | ✅ / 🟡 | Confirm balance handling is documented |
| `01_eda_data_wrangling.ipynb` in the repo | ⬜ | Move the notebook into `notebooks/` |

### 2.2 Audio benchmark and MLOps (milestone 2)

| Requirement | Status | Notes |
|---|---|---|
| 1. Decision Tree | ✅ | Refit in the app from notebook hyperparameters; verify recall |
| 2. SVM, RBF kernel | ✅ | Linear kernel also compared |
| 3. Custom 2D CNN on Mel spectrograms | ✅ | TensorFlow checkpoint (the brief imposes PyTorch) |
| 4. Transfer learning (ResNet18/50 or VGG16) | ⛔ | Optional if time allows |
| 5. Transformer-CNN (AST / Conformer) | ⛔ | Future work |
| Runs logged in MLflow | 🟡 | SVM: logged. Decision Tree and CNN: confirm they are in `audio_benchmark` |
| Select model by recall on the pathological class | 🟡 | Confirm the metric is computed with abnormal as the positive class |
| Export best model to `cardio_model.onnx` | ⬜ | `skl2onnx` for sklearn, `tf2onnx` for the TensorFlow CNN |

### 2.3 Application and API (milestones 5 and 8, partly)

| Requirement | Status | Notes |
|---|---|---|
| FastAPI `/analyze_pcg` accepting a .wav | 🟡 | Confirm whether a FastAPI layer exists or Streamlit calls the models directly |
| Pydantic-validated output | 🟡 | Confirm |
| Inference traced in MLflow | ✅ | Experiment `inference_logs`, link shown in the UI |
| UI to test the 3 models (supervisor request, extra to the brief) | ✅ | Polish items sent to Codex (card overflow, spectrogram axes, disclaimer, SVM grouping) |

### 2.4 Containers (milestone 7)

| Requirement | Status | Notes |
|---|---|---|
| `mlflow-server` container | ✅ | Add volume, `--host 0.0.0.0`, artifact proxy, health check |
| `audio-engine` / API container | ⬜ | |
| UI container | ⬜ | Extra to the brief |
| `docker-compose.yml`, `docker compose up --build` | ⬜ | |
| Multi-stage Dockerfiles | ⬜ | |
| `llm-server`, `vector-db`, `agent-orchestrator` | ⛔ | Not planned |

### 2.5 Testing and CI/CD (milestones 6 and 8)

| Requirement | Status | Notes |
|---|---|---|
| `test_audio.py` (signal processing) | ⬜ | |
| `test_api.py` (HTTPX) | ⬜ | |
| `test_agents.py` | ⛔ | Only if agents are built |
| Coverage with pytest-cov | ⬜ | Keep the report for the PDF |
| `.github/workflows/mlops-ci-cd.yml` | ⬜ | Install, pytest with coverage, build the API image |

### 2.6 LLM, RAG and agents (milestones 3, 4, 5)

| Requirement | Status | Notes |
|---|---|---|
| QLoRA fine-tuning, GGUF export | ⛔ | Needs a GPU |
| RAG over ESC guidelines | ⛔ | Optional stretch |
| LangGraph 3-agent graph, Pydantic guardrails | ⛔ | Optional stretch |

### 2.7 Deliverables

| Deliverable | Status |
|---|---|
| Structured Git repository | 🟡 (folder scaffold created, code to be moved in) |
| `01_eda_data_wrangling.ipynb` | ⬜ |
| `audio_engine/` with training scripts and `cardio_model.onnx` | ⬜ |
| `tests/` | ⬜ |
| `.github/workflows/mlops-ci-cd.yml` | ⬜ |
| `docker/` with Dockerfiles and compose | ⬜ |
| `api/` (`main.py`) | 🟡 |
| Technical report | ⬜ |

---

## 3. Deviations from the brief (declare them in the report)

1. **3 of 5 algorithms.** ResNet/VGG transfer learning and the Transformer-CNN are not implemented. Present them as future work.
2. **TensorFlow CNN**, while the brief lists PyTorch/Torchvision as the imposed stack. Ask the teacher if this is accepted.
3. **Tabular features** are duration, SNR and 20 Mel means, not MFCC/ZCR as described. Explain the choice, and note that `duration` may leak dataset information.
4. **MLflow version** is 3.16.1, while the brief says 2.x.
5. **LLM fine-tuning, RAG, agents and 3 of the 6 services** are out of scope for this submission.
6. **Extra work not in the brief:** an inference UI with per-analysis MLflow logging (supervisor request).

---

## 4. Known risks and limitations to mention

- No held-out test set: results are 3-fold cross-validation estimates, and final models are trained on all samples.
- Models disagree on some files, as expected for a benchmark. Show per-model results, with no combined verdict.
- SVM outputs a decision score, not a probability. Label it clearly and use P(abnormal) for probability-based models.
- The Decision Tree in the app was refit, not loaded from an MLflow artifact. Check that its recall matches the logged value.
- Educational project: not a medical device.

---

## 5. Remaining work in priority order

| # | Task | Estimated effort |
|---|---|---|
| 1 | Verify Decision Tree and CNN runs in MLflow; compare by abnormal-class recall | 0.5 day |
| 2 | Export the winner to `cardio_model.onnx` and check outputs match | 0.5 day |
| 3 | Move code into the repo structure (shared preprocessing/features module) | 1 day |
| 4 | FastAPI `/analyze_pcg` with Pydantic response (if not present) | 1 day |
| 5 | Pytest suite (`test_audio.py`, `test_api.py`) with coverage | 1.5 days |
| 6 | Docker: API and UI containers, compose, MLflow fixes | 2 days |
| 7 | GitHub Actions workflow | 0.5 day |
| 8 | Technical report and screenshots | 2 days |
| 9 | Stretch: ResNet transfer learning, minimal RAG or agent graph | only if time remains |

Estimates are rough. Keep 2 to 3 days of buffer before 1 December.

---

## 6. Questions for the supervisor

1. Is a TensorFlow CNN acceptable, since the brief lists PyTorch?
2. Is implementing 3 of the 5 algorithms acceptable, or is a fourth (ResNet transfer learning) expected?
3. Which location is preferred for `docker-compose.yml`: inside `docker/`, or at the repository root?
4. Should the report include the LLM/RAG/agent parts as design only, or can they be omitted?
