"""Tests for GET /api/v1/videos and GET /api/v1/videos/{video_id}."""

from unittest.mock import patch, MagicMock

SAMPLE_VIDEOS = [
    {"video_id": "v1", "title": "Video One", "channel_id": "c1", "view_count": 100},
    {"video_id": "v2", "title": "Video Two", "channel_id": "c1", "view_count": 200},
]


def _mock_sb(select_data):
    sb = MagicMock()
    select_mock = MagicMock()
    select_mock.execute.return_value = MagicMock(data=select_data)
    select_mock.eq.return_value = select_mock
    select_mock.limit.return_value = select_mock
    sb.table.return_value.select.return_value = select_mock
    return sb


class TestListVideos:
    def test_success(self, client, supabase_env):
        with patch(
            "app.api.v1.endpoints.videos.create_client",
            return_value=_mock_sb(SAMPLE_VIDEOS),
        ):
            resp = client.get("/api/v1/videos")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["count"] == 2
        assert len(body["data"]) == 2

    def test_empty(self, client, supabase_env):
        with patch(
            "app.api.v1.endpoints.videos.create_client", return_value=_mock_sb([])
        ):
            resp = client.get("/api/v1/videos")

        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_missing_env(self, client, no_supabase_env):
        resp = client.get("/api/v1/videos")
        assert resp.status_code == 200
        assert "error" in resp.json()


class TestGetVideo:
    def test_found(self, client, supabase_env):
        row = {
            "video_id": "v1",
            "title": "Video One",
            "transcript": None,
            "created_at": None,
        }
        with patch(
            "app.api.v1.endpoints.videos.create_client", return_value=_mock_sb([row])
        ):
            resp = client.get("/api/v1/videos/v1")

        assert resp.status_code == 200
        assert resp.json()["video_id"] == "v1"

    def test_not_found(self, client, supabase_env):
        with patch(
            "app.api.v1.endpoints.videos.create_client", return_value=_mock_sb([])
        ):
            resp = client.get("/api/v1/videos/nonexistent")

        assert resp.status_code == 404

    def test_missing_env(self, client, no_supabase_env):
        resp = client.get("/api/v1/videos/v1")
        assert resp.status_code == 500
