# ──────────────────────────────────────────────
# Service 5: api-gateway (FastAPI / Uvicorn)
# ──────────────────────────────────────────────
FROM python:3.10-slim AS base

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY audio_engine/ ./audio_engine/
COPY shared/ ./shared/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

