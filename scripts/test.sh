#!/usr/bin/env bash
# Proof-of-function: ensure the semantic-filter LLM (per yt_data_ingestion) is
# available, then run the YouTube data ingestion module.
#
# Usage (from backend root, conda env activated):
#   ./scripts/test.sh
#
# Optional (same as run_local_pipeline for the ingest model):
#   PYTHON=python3 INGEST_LLM_MODEL=gemma2 ./scripts/test.sh
#
# If LLM_PROVIDER=bedrock (in environment or .env), Ollama pull/serve is skipped;
# set AWS + Bedrock model env as usual.

set -euo pipefail

PYTHON="${PYTHON:-python}"

BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BACKEND_ROOT"

# Resolve provider/model the same way as pipelines/yt_data_ingestion._get_llm_provider_for_filtering
# (dotenv, then INGEST_LLM_MODEL overrides LLM_MODEL for this run, default gemma2 + ollama).
read -r LLM_PROVIDER LLM_MODEL < <(
  "$PYTHON" - <<'PY'
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")
p = (os.environ.get("LLM_PROVIDER") or "ollama").lower()
# INGEST_LLM_MODEL matches run_local_pipeline; falls back to LLM_MODEL / default gemma2
m = os.environ.get("INGEST_LLM_MODEL") or os.environ.get("LLM_MODEL") or "gemma2"
print(p, m)
PY
) || {
  echo "error: could not read LLM config (is python-dotenv installed?)" >&2
  exit 1
}

export LLM_PROVIDER
export LLM_MODEL

if [[ "${LLM_PROVIDER}" == "ollama" ]]; then
  ollama_api_ok() {
    curl -sf --connect-timeout 2 "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1
  }

  if ! command -v ollama >/dev/null 2>&1; then
    echo "error: LLM_PROVIDER=ollama but ollama is not in PATH. Install from https://ollama.com" >&2
    exit 1
  fi

  echo "==> Pulling Ollama model for yt_data_ingestion: ${LLM_MODEL}"
  ollama pull "$LLM_MODEL"

  OLLAMA_PID=""
  if ! ollama_api_ok; then
    echo "==> Ollama API not reachable; starting: ollama serve (background)"
    ollama serve &
    OLLAMA_PID=$!
    for _ in $(seq 1 45); do
      if ollama_api_ok; then
        echo "==> Ollama is up."
        break
      fi
      sleep 1
    done
    if ! ollama_api_ok; then
      echo "error: Ollama did not become ready on http://127.0.0.1:11434" >&2
      [[ -n "${OLLAMA_PID}" ]] && kill "$OLLAMA_PID" 2>/dev/null || true
      exit 1
    fi
    cleanup_ollama() {
      if [[ -n "${OLLAMA_PID:-}" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
      fi
    }
    trap cleanup_ollama EXIT
  else
    echo "==> Ollama API already running."
  fi
else
  echo "==> LLM_PROVIDER=${LLM_PROVIDER} (not ollama): skipping Ollama pull/serve."
fi

echo "==> Running pipelines.yt_data_ingestion (LLM_PROVIDER=${LLM_PROVIDER} LLM_MODEL=${LLM_MODEL})"
"$PYTHON" -m pipelines.yt_data_ingestion
