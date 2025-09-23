#!/usr/bin/env bash
# Helper script to run Flask app with AI configuration.

set -euo pipefail

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Change these values to match your environment.
export AI_PROVIDER="local"          # or cloud
export AI_BASE_URL="http://127.0.0.1:8080/v1"
export AI_MODEL="LLaMA_CPP"
export AI_API_KEY="sk-no-key-required"
export AI_TIMEOUT="35"              # override if needed
export AI_CACHE_TTL="120"

LLM_CMD=("./Llama-3.2-3B-Instruct.Q6_K.llamafile" --server --host 127.0.0.1 --port 8080)
"${LLM_CMD[@]}" >/dev/null 2>&1 &
LLM_PID=$!

cleanup() {
  if kill -0 "$LLM_PID" 2>/dev/null; then
    kill "$LLM_PID" >/dev/null 2>&1 || true
    wait "$LLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

sleep 2

PYTHONPATH=src libcamerify python -m pi_kiosk.flask_app --db-path data/kiosk.db --model-dir models --classifier models/face_classifier_70_hash.pkl --camera
