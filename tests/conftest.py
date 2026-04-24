from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from app.main import app

    return TestClient(app)


def _make_execute(data):
    """Return a mock whose .execute().data returns *data*."""
    exe = MagicMock()
    exe.execute.return_value = MagicMock(data=data)
    return exe


def _build_supabase_mock(table_responses: dict[str, list]):
    """
    Build a mock Supabase client where each call to
    `sb.table(name).select(...).execute()` returns the corresponding list.

    Supports chained calls: .eq(), .limit(), .delete(), .upsert(), .insert()
    all pass through and still resolve to .execute().
    """
    sb = MagicMock()

    def _table(name):
        tbl = MagicMock()
        data = table_responses.get(name, [])
        result = MagicMock(data=data)

        # .select(...) -> chainable mock whose .execute() returns data
        select_mock = MagicMock()
        select_mock.execute.return_value = result
        select_mock.eq.return_value = select_mock
        select_mock.limit.return_value = select_mock
        tbl.select.return_value = select_mock

        # .delete() -> chainable mock
        delete_mock = MagicMock()
        delete_mock.eq.return_value = delete_mock
        delete_mock.execute.return_value = result
        tbl.delete.return_value = delete_mock

        # .upsert() / .insert() -> chainable mock
        upsert_mock = MagicMock()
        upsert_mock.execute.return_value = result
        tbl.upsert.return_value = upsert_mock

        insert_mock = MagicMock()
        insert_mock.execute.return_value = result
        tbl.insert.return_value = insert_mock

        return tbl

    sb.table.side_effect = _table
    return sb


@pytest.fixture()
def mock_supabase_factory():
    """Fixture that returns a helper to patch create_client for a given module path."""

    def _factory(module_path: str, table_responses: dict[str, list]):
        sb = _build_supabase_mock(table_responses)
        patcher = patch(f"{module_path}.create_client", return_value=sb)
        mock_create = patcher.start()
        return sb, mock_create, patcher

    return _factory


@pytest.fixture()
def supabase_env(monkeypatch):
    """Ensure Supabase env vars are set so endpoints don't 503."""
    monkeypatch.setattr(
        "app.core.config.settings.SUPABASE_URL", "https://fake.supabase.co"
    )
    monkeypatch.setattr(
        "app.core.config.settings.SUPABASE_SERVICE_ROLE_KEY", "fake-key"
    )


@pytest.fixture()
def no_supabase_env(monkeypatch):
    """Clear Supabase env vars to test 503 / error paths."""
    monkeypatch.setattr("app.core.config.settings.SUPABASE_URL", "")
    monkeypatch.setattr("app.core.config.settings.SUPABASE_SERVICE_ROLE_KEY", "")
