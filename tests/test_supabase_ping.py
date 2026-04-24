"""Tests for GET /api/v1/supabase/ping."""

from unittest.mock import patch, MagicMock


def test_ping_success(client, supabase_env):
    mock_sb = MagicMock()
    select_mock = MagicMock()
    select_mock.limit.return_value = select_mock
    select_mock.execute.return_value = MagicMock(data=[{"video_id": "abc"}])
    mock_sb.table.return_value.select.return_value = select_mock

    with patch(
        "app.api.v1.endpoints.supabase_ping.create_client", return_value=mock_sb
    ):
        resp = client.get("/api/v1/supabase/ping")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["data"] == [{"video_id": "abc"}]


def test_ping_missing_env(client, no_supabase_env):
    resp = client.get("/api/v1/supabase/ping")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "error" in body


def test_ping_exception(client, supabase_env):
    with patch(
        "app.api.v1.endpoints.supabase_ping.create_client",
        side_effect=Exception("connection refused"),
    ):
        resp = client.get("/api/v1/supabase/ping")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "connection refused" in body["error"]
