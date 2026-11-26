# ──────────────────────────────────────────────
# Service 1: llm-server (Ollama + QLoRA/GGUF)
# ──────────────────────────────────────────────
FROM ollama/ollama:latest

# Copy the Modelfile and startup script
COPY docker/llm-server/Modelfile /root/Modelfile
COPY docker/llm-server/entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/root/entrypoint.sh"]

