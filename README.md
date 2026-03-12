# YouTube Intelligence Platform — Backend
## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # fill in your Supabase credentials
```

## Development Setup

```bash
# Install dev dependencies and configure git hooks
bash scripts/setup-hooks.sh
```

Or manually:

```bash
# Install dev dependencies
uv pip install -r requirements-dev.txt  # or: pip install -r requirements-dev.txt

# Configure git hooks
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

The pre-commit hook runs **ruff format** (formatting) and **ruff check** (linting) on staged Python files.

## Database Migrations (Alembic)

Migrations live in `alembic/versions/`. The database URL is read from `DATABASE_URL` in `.env`.

```bash
# Apply all pending migrations
uv run alembic upgrade head

# Generate a new migration after changing models in app/models/
uv run alembic revision --autogenerate -m "describe your change"

# Check current migration version
uv run alembic current

# Rollback the last migration
uv run alembic downgrade -1
```

## Run the server

```bash
uv run uvicorn app.main:app --reload
```

## Data Ingestion Pipeline

### Setup
1. Go to Google Console >> Get an API key for YouTube Data API
2. Initialize a `.env` file with `YOUTUBE_DATA_API_KEY` set up. 
3. Create a designated virtual env (either via Python natively or Anaconda) and activate it. 
4. Install the following packages/libraries using the following command:
    ```bash
    pip install google-api-python-client youtube-transcript-api python-dotenv
    ```
5. Run the notebook (`yt-data-ingestion.ipynb`) to see the full pipeline in action.

### LLM Insight Generation (Claims Extraction)

The `llm_insight_generation.ipynb` notebook extracts claims from cleaned transcripts using Ollama.

### Ollama Setup

1. **Install Ollama**: Download from [ollama.com](https://ollama.com) or run `brew install ollama` (macOS).
2. **Start Ollama**: Open the Ollama app (macOS) or run `ollama serve` in a terminal.
3. **Pull a model** (required before running the notebook):
   ```bash
   ollama pull llama3
   ```
   Or use another model (e.g. `llama3.2`, `mistral`).
4. **Optional – use a different model**: Add `LLM_MODEL=llama3.2` to your `.env` (or export it). Default is `llama3`.

### Running the Claims Pipeline

1. Ensure Ollama is running and at least one model is pulled (`ollama list`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run all cells in `pipelines/llm_insight_generation.ipynb` from the top.
4. The test cell extracts claims from `data/transcripts/cleaned/gpzDxm7qflY.txt`.
