#!/usr/bin/env bash
# Run local pipelines in order: ingestion → LLM insights → sentiment → misinfo.
# Ensures Ollama is used for steps 1–2 (pull models + optional ollama serve).
#
# Usage (from anywhere):
#   chmod +x scripts/run_local_pipeline.sh
#   ./scripts/run_local_pipeline.sh
#
# Recommended: activate your env first, e.g. conda activate yt-intel-project
#
# Optional env overrides:
#   PYTHON=python3              Python to use (default: python)
#   INGEST_LLM_MODEL=gemma2    Model for yt_data_ingestion semantic filter
#   INSIGHT_LLM_MODEL=qwen3    Model for llm_insight_generation
#   SENTIMENT_MODE=2          sentiment_analysis mode (default: 2 = all videos in Supabase)
#   MISINFO_MODE=2            misinfo_checker mode (default: 2 = all videos in Supabase)

set -euo pipefail

PYTHON="${PYTHON:-python}"

BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BACKEND_ROOT"

# --- Ollama + LLM env (steps 1–2) -------------------------------------------
# Defaults match pipeline code: yt_data_ingestion uses gemma2; llm_insight_generation uses qwen3.
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"

INGEST_LLM_MODEL="${INGEST_LLM_MODEL:-gemma2}"
INSIGHT_LLM_MODEL="${INSIGHT_LLM_MODEL:-qwen3}"

# sentiment_analysis / misinfo_checker CLI modes (see each module's __main__)
SENTIMENT_MODE="${SENTIMENT_MODE:-2}"
MISINFO_MODE="${MISINFO_MODE:-2}"

ollama_api_ok() {
  curl -sf --connect-timeout 2 "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1
}

if ! command -v ollama >/dev/null 2>&1; then
  echo "error: ollama not found in PATH. Install from https://ollama.com" >&2
  exit 1
fi

echo "==> Pulling Ollama models for this run (${INGEST_LLM_MODEL}, ${INSIGHT_LLM_MODEL})..."
ollama pull "$INGEST_LLM_MODEL"
ollama pull "$INSIGHT_LLM_MODEL"

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
    [[ -n "$OLLAMA_PID" ]] && kill "$OLLAMA_PID" 2>/dev/null || true
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

run_py() {
  "$PYTHON" "$@"
}

echo "==> [1/4] yt_data_ingestion (Ollama, model=${INGEST_LLM_MODEL})"
LLM_MODEL="$INGEST_LLM_MODEL" run_py -m pipelines.yt_data_ingestion

echo "==> [2/4] llm_insight_generation (Ollama, model=${INSIGHT_LLM_MODEL})"
LLM_MODEL="$INSIGHT_LLM_MODEL" run_py -m pipelines.llm_insight_generation

echo "==> [3/4] sentiment_analysis (mode ${SENTIMENT_MODE})"
run_py -m pipelines.sentiment_analysis "$SENTIMENT_MODE"

echo "==> [4/4] misinfo_checker (mode ${MISINFO_MODE})"
run_py -m pipelines.misinfo_checker "$MISINFO_MODE"

echo "==> All steps finished."
