# ──────────────────────────────────────────────
# Service 4: agent-orchestrator (LangGraph)
# ──────────────────────────────────────────────
FROM python:3.10-slim AS base

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir \
       langgraph>=0.0.40 \
       langchain>=0.1 \
       langchain-community>=0.0.10 \
       qdrant-client>=1.7 \
       sentence-transformers>=2.2 \
       httpx>=0.27

COPY agents/ ./agents/
COPY rag_engine/ ./rag_engine/
COPY shared/ ./shared/

EXPOSE 8004

CMD ["uvicorn", "agents.orchestrator.server:app", "--host", "0.0.0.0", "--port", "8004"]

