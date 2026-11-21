#!/bin/sh
# Writes /config.js at container start so API_URL / MLFLOW_PUBLIC_URL can change without a rebuild.
set -eu
cat > /usr/share/nginx/html/config.js <<CONFIG
window.__APP_CONFIG__ = { apiUrl: "${API_URL:-}", mlflowPublicUrl: "${MLFLOW_PUBLIC_URL:-}" };
CONFIG
