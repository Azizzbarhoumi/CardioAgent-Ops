# ──────────────────────────────────────────────
# Service 2: vector-db (Qdrant)
# ──────────────────────────────────────────────
FROM qdrant/qdrant:latest

COPY docker/vector-db/config.yaml /qdrant/config/production.yaml

EXPOSE 6333 6334

