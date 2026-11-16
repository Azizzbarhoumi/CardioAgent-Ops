FROM python:3.12-slim AS builder

WORKDIR /build
COPY docker/frontend/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10002 app

WORKDIR /app
COPY --from=builder /install /usr/local
COPY frontend ./frontend
COPY audio_engine/__init__.py ./audio_engine/__init__.py
COPY audio_engine/features ./audio_engine/features
COPY audio_engine/preprocessing ./audio_engine/preprocessing

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
