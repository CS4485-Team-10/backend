"""Tests for POST /api/v1/ingest/video."""

from unittest.mock import patch, MagicMock

import httpx

PIPELINE_RESULT = {
    "channel": {
        "channel_id": "ch1",
        "title": "Test Channel",
        "handle": "@test",
        "url": "https://www.youtube.com/channel/ch1",
    },
    "video": {
        "video_id": "v1",
        "channel_id": "ch1",
        "title": "Test Video",
        "description": "desc",
        "view_count": 1000,
        "published_at": "2026-01-01T00:00:00Z",
        "thumbnail_url": None,
        "stats": {},
    },
    "transcript": {
        "video_id": "v1",
        "language": "en",
        "content": "Hello world",
    },
}


def _mock_sb():
    sb = MagicMock()
    result = MagicMock(data=[])
    upsert = MagicMock()
    upsert.execute.return_value = result
    sb.table.return_value.upsert.return_value = upsert
    sb.table.return_value.insert.return_value = upsert
    delete_mock = MagicMock()
    delete_mock.eq.return_value = delete_mock
    delete_mock.execute.return_value = result
    sb.table.return_value.delete.return_value = delete_mock
    return sb


class TestIngestVideo:
    def test_success(self, client, supabase_env, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.YOUTUBE_API_KEY", "fake-key")

        with (
            patch(
                "app.api.v1.endpoints.ingest.run_pipeline", return_value=PIPELINE_RESULT
            ),
            patch("app.api.v1.endpoints.ingest.create_client", return_value=_mock_sb()),
        ):
            resp = client.post("/api/v1/ingest/video", json={"video_id": "v1"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["video_id"] == "v1"
        assert body["channel_id"] == "ch1"

    def test_empty_video_id(self, client, supabase_env):
        resp = client.post("/api/v1/ingest/video", json={"video_id": ""})
        assert resp.status_code == 422

    def test_missing_body(self, client, supabase_env):
        resp = client.post("/api/v1/ingest/video", json={})
        assert resp.status_code == 422

    def test_missing_supabase(self, client, no_supabase_env, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.YOUTUBE_API_KEY", "fake-key")
        resp = client.post("/api/v1/ingest/video", json={"video_id": "v1"})
        assert resp.status_code == 503

    def test_missing_youtube_key(self, client, supabase_env, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.YOUTUBE_API_KEY", "")
        monkeypatch.setattr("app.core.config.settings.YOUTUBE_DATA_API_KEY", "")
        resp = client.post("/api/v1/ingest/video", json={"video_id": "v1"})
        assert resp.status_code == 503

    def test_youtube_http_error(self, client, supabase_env, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.YOUTUBE_API_KEY", "fake-key")

        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch(
            "app.api.v1.endpoints.ingest.run_pipeline",
            side_effect=httpx.HTTPStatusError(
                "forbidden", request=MagicMock(), response=mock_response
            ),
        ):
            resp = client.post("/api/v1/ingest/video", json={"video_id": "v1"})

        assert resp.status_code == 403

    def test_video_not_found(self, client, supabase_env, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.YOUTUBE_API_KEY", "fake-key")

        with patch(
            "app.api.v1.endpoints.ingest.run_pipeline",
            side_effect=ValueError("Video not found"),
        ):
            resp = client.post("/api/v1/ingest/video", json={"video_id": "v1"})

        assert resp.status_code == 404
