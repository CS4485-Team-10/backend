# YouTube Intelligence Platform — Backend

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # fill in your Supabase credentials
```

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
