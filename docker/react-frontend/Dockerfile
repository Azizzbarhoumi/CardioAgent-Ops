# React frontend: multi-stage build (node build -> nginx serve). Build context: ./frontend
FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Build-time defaults; both can be overridden at RUN time with API_URL / MLFLOW_PUBLIC_URL (see 40-runtime-config.sh)
ARG VITE_API_URL=http://localhost:8000
ARG VITE_MLFLOW_PUBLIC_URL=http://localhost:5000
ENV VITE_API_URL=${VITE_API_URL} VITE_MLFLOW_PUBLIC_URL=${VITE_MLFLOW_PUBLIC_URL}
RUN npm run build

FROM nginx:1.29-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY --from=dockerconf nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=dockerconf 40-runtime-config.sh /docker-entrypoint.d/40-runtime-config.sh
RUN chmod +x /docker-entrypoint.d/40-runtime-config.sh
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
