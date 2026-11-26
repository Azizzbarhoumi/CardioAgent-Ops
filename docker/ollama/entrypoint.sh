#!/bin/bash
# entrypoint.sh — Start Ollama, pull base model, register CardioAgent model
set -e

echo ">>> Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for the server to be ready
echo ">>> Waiting for Ollama to be ready..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done

echo ">>> Pulling base model (this may take a while on first run)..."
ollama pull qwen2.5:7b-instruct || true

echo ">>> Creating CardioAgent custom model from Modelfile..."
ollama create cardioagent -f /root/Modelfile || true

echo ">>> Ollama is ready. CardioAgent model registered."
wait $SERVER_PID

