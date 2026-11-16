FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir "mlflow==3.16.1"

WORKDIR /mlflow
RUN mkdir -p /mlflow/artifacts

EXPOSE 5000
HEALTHCHECK --interval=20s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5)" || exit 1

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", "--allowed-hosts", "mlflow,mlflow:5000,mlflow-server,mlflow-server:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000", "--host", "0.0.0.0", "--port", "5000"]
