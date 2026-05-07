# YouTube Intelligence Platform — Backend

Backend for the **YouTube Intelligence Platform**: it ingests YouTube data, extracts health-related claims with an LLM, groups claims into narratives, and exposes the results through a versioned REST API.

## What This Backend Does

- Pulls YouTube metadata and transcripts into Supabase
- Filters and prioritizes content for public-health relevance (including semantic filtering and impact-style signals in batch flows).
- Uses an LLM to extract generalizable **health-related claims** from transcripts.
- Matches claims to **narratives** using embeddings and creates new narratives when needed.
- Serves **versioned API endpoints** (`/api/v1/...`) so clients can query health, overview, claims, narratives, ingestion, and related resources.

## Architecture Overview

Requests hit a **FastAPI** service. It reads and writes from **Supabase** via SQLModel/SQLAlchemy. **Ingestion pipelines** fetch YouTube data (Data API + transcripts) and upsert into core tables. A separate **LLM pipeline** extracts claims and runs **narrative matching** (embedding similarity, backed by Amazon Bedrock Titan embeddings in the default setup). **Deployment** can be configured via`render.yaml`.

## Tech Stack


| Area                         | Technologies                                                                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| API & runtime                | FastAPI, Uvicorn                                                                                                |
| DB & Schema                  | Supabase (Postgres), SQLAlchemy 2.x, SQLModel                                                                   |
| Migrations                   | Alembic (`alembic/`)                                                                                            |
| YouTube Data Ingestion       | YouTube Data API v3, `youtube-transcript-api`                                                                   |
| Large Language Models (LLMs) | Switchable **Ollama / Amazon Bedrock** (environment-driven)                                                     |
| Narrative Embeddings         | Amazon Bedrock - Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) via `bedrock-runtime` `invoke_model` |
| Quality of life              | Ruff (format + lint), `python-dotenv`, Pydantic settings                                                        |


## Repository Structure

- `**app/`** — The live API: routers under `/api/v1`, configuration, database sessions, SQLModel tables, and request/response schemas. Pipeline helpers used by HTTP endpoints (for example, single-video ingest) live here under `app/pipelines/`.
- `pipelines/` — Batch jobs you run locally or can be configured to run async via AWS Lambda: YouTube search + filtering + persistence, LLM claim extraction and narrative linking, embedding-based matching helpers, and shared pipeline utilities.
- `**alembic/`** — Database migrations (`versions/`) and Alembic runtime configuration (`env.py`).
- `**scripts/**` — Developer utilities (for example, git hook setup and local orchestration scripts).

## Installation & Setup

From `backend/`, create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional formatting, linting, and pre-commit hooks:

```bash
pip install -r requirements-dev.txt
bash scripts/setup-hooks.sh
```

Format and lint with Ruff:

```bash
ruff format .
ruff check .
```

If you ran `scripts/setup-hooks.sh`, Ruff also runs on staged files via the pre-commit hook.

## Running the Project

**API server** — Starts the FastAPI app with auto-reload for local development:

```bash
uvicorn app.main:app --reload
```

Health check: `GET /api/v1/health`

---

**Ingest a single video (via API)** — Fetches video, channel, and transcript data and upserts into Supabase (`channels`, `videos`, `transcripts`):

- `POST /api/v1/ingest/video` with JSON `{ "video_id": "<id>" }`  
(Implementation uses `app/pipelines/yt_ingest.py`.)

---

**Batch ingest + filter** — Searches YouTube, applies an LLM semantic filter for public-health relevance, applies impact-style filtering, and persists videos and transcripts to Supabase:

```bash
python -m pipelines.yt_data_ingestion
```

Example provider overrides:

```bash
# local test
LLM_PROVIDER=ollama LLM_MODEL=gemma2 python -m pipelines.yt_data_ingestion

# cloud runtime
LLM_PROVIDER=bedrock LLM_MODEL=gemma2 python -m pipelines.yt_data_ingestion
```

---

**Claims + narratives (LLM insight generation)** — Reads transcripts without claims yet, extracts health-related claims, matches them to narratives with embeddings (default: Amazon Bedrock Titan Text Embeddings V2), opens new narratives when appropriate, and writes `claims`, `narratives`, and `claim_narratives`:

```bash
python -m pipelines.llm_insight_generation
```

## Environment Variables

Create a local `.env` (never commit it). The app loads it through Pydantic settings and `python-dotenv`.

**Required for typical local development**


| Variable                                    | Purpose                                  |
| ------------------------------------------- | ---------------------------------------- |
| `DATABASE_URL`                              | Postgres connection for SQLModel/Alembic |
| `SUPABASE_URL`                              | Supabase project URL                     |
| `SUPABASE_SERVICE_ROLE_KEY`                 | Service role access for Supabase         |
| `YOUTUBE_DATA_API_KEY` or `YOUTUBE_API_KEY` | YouTube Data API v3                      |


**Required for narrative matching / embeddings (default Bedrock path)**


| Variable                                     | Purpose                  |
| -------------------------------------------- | ------------------------ |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | AWS credentials          |
| `AWS_REGION` or `AWS_DEFAULT_REGION`         | Region for Bedrock calls |


**Common optional**


| Variable                                                                                               | Notes                                  |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| `FRONTEND_URL`, `PORT`, `ENV`                                                                          | App/runtime tuning                     |
| `LLM_PROVIDER`                                                                                         | Default `ollama`; switch with env      |
| `LLM_MODEL`, `OLLAMA_BASE_URL`                                                                         | Model and Ollama endpoint              |
| `YT_QUOTA_DAILY_BUDGET_UNITS`                                                                          | Caps YouTube quota usage               |
| `NARR_EMBEDDING_BACKEND`                                                                               | Default `bedrock`                      |
| `NARR_EMBEDDING_MODEL`                                                                                 | Default `amazon.titan-embed-text-v2:0` |
| `NARR_EMBEDDING_DIMENSIONS`                                                                            | Default `512`                          |
| `NARR_EMBEDDING_NORMALIZE`                                                                             | Default `true`                         |
| `NARR_EMBEDDING_TIMEOUT`, `NARR_STRONG_MATCH`, `NARR_MULTI_LINK`, `NARR_NEW_MIN`, `NARR_MAX_PER_CLAIM` | Matching tuning knobs                  |




## Deployment Notes

- `render.yaml` defines a basic Render deployment for the FastAPI service



## Developer Notes

### LLM Provider Mechanisms

- `pipelines/yt_data_ingestion.py` uses `LLM_PROVIDER` + `LLM_MODEL` (`ollama` or `bedrock`). If `LLM_MODEL` is unset, ingestion defaults to `gemma2`.
- `pipelines/llm_insight_generation.py` uses the same variables; if `LLM_MODEL` is unset, it defaults to `qwen3`.

### Cloud Console Credentials

- **Google Cloud**: enable *YouTube Data API v3*, create an API key, set `YOUTUBE_DATA_API_KEY`.
- **AWS**: in your chosen region, request access to `amazon.titan-embed-text-v2:0` in the Bedrock model catalog and ensure the IAM principal can `bedrock:InvokeModel` for that model.

### DB Migrations

- Migrations live in `alembic/versions/`. Alembic reads `DATABASE_URL` from `alembic/env.py`; `alembic.ini` intentionally omits the URL.

```bash
alembic upgrade head
alembic current
alembic revision --autogenerate -m "describe your change"
```





### API Layouts (`app/`)

For contributors who want a quick map of the API package:

- `app/main.py` — FastAPI app, CORS, routers mounted at `/api/v1`
- `app/api/v1/endpoints/` — Endpoints such as health, overview, claims, narratives, ingest, etc.
- `app/core/config.py` — Settings + `.env` loading
- `app/core/database.py` — Database sessions for routes
- `app/models/` — SQLModel models (videos, claims, narratives, joins, etc.)
- `app/schemas/` — Pydantic request/response shapes

### Data Pipeline Layouts (`pipelines/`)

- `pipelines/yt_data_ingestion.py` — Search + semantic filter + impact filter + transcript extraction
- `pipelines/llm_insight_generation.py` — Claims, narratives extraction
- `pipelines/narrative_matching.py` — Embedding + cosine similarity (Bedrock Titan Text Embeddings V2 via `invoke_model`)
- `pipelines/shared/` — Shared helpers, interfaces and dataclasses



### Related Repositories

- [YouTube Intelligence Platform — Frontend](https://github.com/CS4485-Team-10/frontend)

