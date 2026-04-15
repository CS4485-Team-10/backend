# YouTube Intelligence Platform — Backend

Backend services and pipelines for ingesting YouTube content, extracting health-related claims, matching them into narratives, and serving data via a FastAPI API.

## Overview

This repository contains:

- **FastAPI service** under `app/` with versioned REST endpoints (`/api/v1/...`).
- **Database models + migrations** (SQLModel/SQLAlchemy + Alembic) for the core analytics schema.
- **Pipelines** for:
  - ingesting YouTube metadata + transcripts into Supabase
  - extracting claims via an LLM provider (local Ollama by default)
  - matching claims to narratives using embeddings (Google Gemini `embedContent`)
- **Scripts** in `scripts/` for running pipeline utilities locally.

## Tech Stack

- **API**: FastAPI + Uvicorn
- **DB/ORM & Migrations**: Supabase, SQLAlchemy 2.x, SQLModel, Alembic (`alembic/`)
- **YouTube Data Ingestion**: YouTube Data API v3 + `youtube-transcript-api`
- **LLM inference**: Ollama (OpenAI-compatible endpoint) with optional provider abstraction
- **Embeddings**: Google Gemini `embedContent` (remote HTTP)
- **Tooling**: Ruff (format + lint), `python-dotenv`

## Setup

### Python environment (venv)

From `backend/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional dev tooling (Ruff + hooks):

```bash
pip install -r requirements-dev.txt
bash scripts/setup-hooks.sh
```

### Ruff (format + lint)

```bash
ruff format .
ruff check .
```

If you ran `scripts/setup-hooks.sh`, Ruff also runs automatically on staged files via the pre-commit hook.

### Environment variables (`.env`)

Create a local `.env` (do not commit). This repo loads it via Pydantic settings and `python-dotenv`.

- **Required (local dev)**
  - `DATABASE_URL`
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`
  - `YOUTUBE_DATA_API_KEY` (or `YOUTUBE_API_KEY`)
- **Required (narrative matching / embeddings)**
  - `NARR_EMBEDDING_BACKEND=remote`
  - `NARR_EMBEDDING_URL` (Gemini `...:embedContent`)
  - `NARR_EMBEDDING_API_KEY`
- **Common optional**
  - `FRONTEND_URL`, `PORT`, `ENV`
  - `LLM_PROVIDER` (default `ollama`), `LLM_MODEL`, `OLLAMA_BASE_URL`
  - `YT_SEMANTIC_FILTER_MODEL`, `YT_QUOTA_DAILY_BUDGET_UNITS`
  - `NARR_EMBEDDING_TIMEOUT`, `NARR_STRONG_MATCH`, `NARR_MULTI_LINK`, `NARR_NEW_MIN`, `NARR_MAX_PER_CLAIM`

Google Console:

- **YouTube**: enable *YouTube Data API v3* → create API key → set `YOUTUBE_DATA_API_KEY`
- **Gemini**: create Gemini/Google AI API key → set `NARR_EMBEDDING_API_KEY` (+ `NARR_EMBEDDING_URL`)

### Database migrations (Alembic)

Migrations live in `alembic/versions/`. Alembic reads the DB URL from `DATABASE_URL` (wired in `alembic/env.py`; `alembic.ini` is intentionally URL-less).

```bash
alembic upgrade head
alembic current
alembic revision --autogenerate -m "describe your change"
```

## Running

### Run the API server

```bash
uvicorn app.main:app --reload
```

Health check:

- `GET /api/v1/health`

### Run pipelines locally

#### Ingest a single video (API route)

- `POST /api/v1/ingest/video` with JSON `{ "video_id": "<id>" }`
  - Uses `app/pipelines/yt_ingest.py` to fetch video/channel/transcript and upserts to Supabase tables (`channels`, `videos`, `transcripts`).

#### Batch ingest + filter (script entrypoint)

```bash
python -m pipelines.yt_data_ingestion
```

This pipeline searches YouTube, applies an LLM semantic filter (public-health relevance), filters by impact metrics, and persists `videos` + `transcripts` to Supabase.

#### LLM insight generation (claims + narratives)

```bash
python -m pipelines.llm_insight_generation
```

This reads Supabase transcripts that don’t yet have claims, extracts generalizable health-related claims, matches them to existing narratives via embeddings (Gemini), creates new narratives when needed, and writes back to Supabase (`claims`, `narratives`, `claim_narratives`).

## Architecture

### `app/` (FastAPI application)

- `**app/main.py**`: FastAPI app, CORS, router mounting at `/api/v1`
- `**app/api/**`: versioned API routes
  - `app/api/v1/endpoints/`: endpoints such as `health`, `overview`, `claims`, `narratives`, `ingest`, etc.
- `**app/core/**`: application config and infrastructure
  - `app/core/config.py`: Pydantic settings; loads `.env`
  - `app/core/database.py`: DB session wiring (used by API endpoints)
- `**app/models/**`: SQLModel models representing DB tables (videos, claims, narratives, join tables, etc.)
- `**app/schemas/**`: Pydantic response/request schemas for API responses
- `**app/pipelines/**`: API-facing pipeline helpers (e.g. `yt_ingest.py` used by `/ingest/video`)

### `pipelines/` (batch + ML/LLM pipelines)

Standalone pipeline modules (typically run via `python -m ...`):

- `**pipelines/yt_data_ingestion.py**`: YouTube search + semantic filter + impact filter + persist to Supabase
- `**pipelines/llm_insight_generation.py**`: claim extraction + narrative creation/linking + persist to Supabase
- `**pipelines/narrative_matching.py**`: embedding + cosine similarity matching logic (Gemini `embedContent`)
- `**pipelines/shared/**`: shared pipeline utilities / interfaces

### `alembic/` (migrations)

- `**alembic/env.py**`: migration runtime config (loads `DATABASE_URL`)
- `**alembic/versions/**`: migration revisions

### `scripts/` (developer utilities)

Convenience scripts for running parts of the pipeline locally:

- `scripts/run_pipeline.py`: runs selected pipeline scripts
- `scripts/setup-hooks.sh`: installs dev deps + enables git hooks
- Other one-off analysis scripts (`sentiment_analysis.py`, `misinfo_checker.py`, etc.)

## Deployment notes

- `render.yaml` contains a basic Render configuration for running the FastAPI service.
- Ensure all required secrets are configured as environment variables in the deployment environment (do not rely on a checked-in `.env`).

