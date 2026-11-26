#!/bin/bash
# Start Ollama, pull the base model, register the CardioAgent model.
set -e

echo ">>> Starting Ollama server..."
ollama serve &
SERVER_PID=$!

echo ">>> Waiting for Ollama to be ready..."
until ollama list > /dev/null 2>&1; do sleep 1; done

echo ">>> Pulling base model..."
ollama pull qwen2.5:7b-instruct || true

echo ">>> Creating CardioAgent custom model..."
ollama create cardioagent -f /root/Modelfile || true

echo ">>> Ollama ready. CardioAgent model registered."
wait $SERVER_PID

