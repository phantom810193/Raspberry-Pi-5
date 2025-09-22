#!/usr/bin/env bash
# Helper script to run Flask app with AI configuration.

set -euo pipefail

# Change these values to match your environment.
export AI_PROVIDER="local"          # or cloud
export AI_BASE_URL="http://127.0.0.1:8080/v1"
export AI_MODEL="LLaMA_CPP"
export AI_API_KEY="sk-no-key-required"
export AI_TIMEOUT="35"              # override if needed
export AI_CACHE_TTL="60"

PYTHONPATH=src python -m pi_kiosk.flask_app --camera --db-path data/kiosk.db --model-dir models "$@"
