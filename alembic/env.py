from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

from alembic import context

from app.core.config import settings
import app.models  # noqa: F401 – registers all models

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# region agent log
def _agent_log_alembic_env(hypothesis_id: str, message: str, data: dict) -> None:
    import json
    import time
    from pathlib import Path

    try:
        log_path = Path(__file__).resolve().parent.parent / ".cursor" / "debug-5c359c.log"
        entry = {
            "sessionId": "5c359c",
            "runId": "initial",
            "hypothesisId": hypothesis_id,
            "location": "alembic/env.py:16",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Logging must never break migrations
        pass


_agent_log_alembic_env(
    hypothesis_id="H1",
    message="DATABASE_URL characteristics before set_main_option",
    data={
        "is_set": bool(settings.DATABASE_URL),
        "length": len(settings.DATABASE_URL or ""),
        "contains_percent": "%" in (settings.DATABASE_URL or ""),
    },
)
# endregion agent log

config.set_main_option(
    "sqlalchemy.url",
    settings.DATABASE_URL.replace("%", "%%") if settings.DATABASE_URL else "",
)

target_metadata = SQLModel.metadata

# Tables managed outside Alembic (pre-existing in Supabase)
EXCLUDE_TABLES = {"jobs", "trend_points"}


def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and name in EXCLUDE_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
