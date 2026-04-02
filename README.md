# YouTube Intelligence Platform — Backend

## Setup (Conda)

```bash
conda create -n yt-intel-project python=3.12 -y
conda activate yt-intel-project
pip install -r requirements.txt
# Create `.env` with DATABASE_URL, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, etc.
```

Fill in `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and other keys your app needs in `.env`.

### Alternative: uv + `.venv`

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Development Setup

```bash
pip install -r requirements-dev.txt
bash scripts/setup-hooks.sh
```

Or manually:

```bash
pip install -r requirements-dev.txt
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

The pre-commit hook runs **ruff format** (formatting) and **ruff check** (linting) on staged Python files.

## Database Migrations (Alembic)

Migrations live in `alembic/versions/`. The database URL is read from `DATABASE_URL` in `.env`.

```bash
# Apply all pending migrations
alembic upgrade head

# Generate a new migration after changing models in app/models/
alembic revision --autogenerate -m "describe your change"

# Check current migration version
alembic current

# Rollback the last migration
alembic downgrade -1
```

If you use `uv`, prefix commands with `uv run` (for example `uv run alembic upgrade head`).

## Run the server

```bash
uvicorn app.main:app --reload
```

(With `uv`: `uv run uvicorn app.main:app --reload`.)

## Data Ingestion Pipeline

### Setup

1. In [Google Cloud Console](https://console.cloud.google.com/), enable the YouTube Data API and create an API key.
2. Set `YOUTUBE_DATA_API_KEY` in `.env`.
3. Activate the **`yt-intel-project`** conda env (or your venv) and install dependencies: `pip install -r requirements.txt` (includes `google-api-python-client`, `youtube-transcript-api`, `python-dotenv`, and the rest of the backend stack).
4. Run `yt-data-ingestion.ipynb` or import `pipelines.yt_data_ingestion` to run the pipeline in code.

### LLM Insight Generation (Claims Extraction)

The `llm_insight_generation.ipynb` notebook (and `pipelines/llm_insight_generation.py`) extract claims from cleaned transcripts using Ollama.

### Ollama Setup

1. **Install Ollama**: [ollama.com](https://ollama.com) or `brew install ollama` (macOS).
2. **Start Ollama**: Open the Ollama app or run `ollama serve`.
3. **Pull the models** used by this repo:
   ```bash
   ollama pull gemma2
   ollama pull qwen3
   ```
4. **Model selection (defaults)**  
   - **Claims / insight pipeline** (`llm_insight_generation`): default **`qwen3`**. Override with **`LLM_MODEL`** in `.env`.  
   - **YouTube ingestion — public-health semantic filter** (`yt_data_ingestion`): default **`gemma2`**. Override with **`YT_SEMANTIC_FILTER_MODEL`** (this is separate from `LLM_MODEL` so insight and ingestion can use different models in the same `.env`).

### Running the Claims Pipeline

1. Ensure Ollama is running and the models you need are pulled (`ollama list`).
2. Dependencies: `pip install -r requirements.txt` (inside `yt-intel-project` or equivalent).
3. Run all cells in `pipelines/llm_insight_generation.ipynb` from the top.
4. The test cell extracts claims from `data/transcripts/cleaned/gpzDxm7qflY.txt`.
