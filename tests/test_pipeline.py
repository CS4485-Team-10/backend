"""Tests for app.pipelines.yt_ingest functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.pipelines.yt_ingest import (
    fetch_channel_metadata,
    fetch_video_metadata,
    run_pipeline,
)

VIDEO_API_RESPONSE = {
    "items": [
        {
            "snippet": {
                "title": "Test Video",
                "description": "A description",
                "channelId": "ch1",
                "publishedAt": "2026-01-01T00:00:00Z",
                "thumbnails": {"default": {"url": "https://img.youtube.com/thumb.jpg"}},
            },
            "statistics": {"viewCount": "12345"},
            "contentDetails": {},
        }
    ]
}

CHANNEL_API_RESPONSE = {
    "items": [
        {
            "snippet": {
                "title": "Test Channel",
                "customUrl": "@testchannel",
            }
        }
    ]
}


def _mock_httpx_get(json_data, status_code=200):
    """Return a mock response for httpx.Client.get."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


class TestFetchVideoMetadata:
    def test_success(self):
        mock_resp = _mock_httpx_get(VIDEO_API_RESPONSE)
        with patch("app.pipelines.yt_ingest.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(
                return_value=MagicMock(get=MagicMock(return_value=mock_resp))
            )
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = fetch_video_metadata("fake-key", "v1")

        assert result["snippet"]["title"] == "Test Video"
        assert result["statistics"]["viewCount"] == "12345"

    def test_video_not_found(self):
        mock_resp = _mock_httpx_get({"items": []})
        with patch("app.pipelines.yt_ingest.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(
                return_value=MagicMock(get=MagicMock(return_value=mock_resp))
            )
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(ValueError, match="Video not found"):
                fetch_video_metadata("fake-key", "nonexistent")

    def test_http_error(self):
        mock_resp = _mock_httpx_get({}, status_code=403)
        with patch("app.pipelines.yt_ingest.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(
                return_value=MagicMock(get=MagicMock(return_value=mock_resp))
            )
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(httpx.HTTPStatusError):
                fetch_video_metadata("fake-key", "v1")


class TestFetchChannelMetadata:
    def test_success(self):
        mock_resp = _mock_httpx_get(CHANNEL_API_RESPONSE)
        with patch("app.pipelines.yt_ingest.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(
                return_value=MagicMock(get=MagicMock(return_value=mock_resp))
            )
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = fetch_channel_metadata("fake-key", "ch1")

        assert result["title"] == "Test Channel"
        assert result["handle"] == "@testchannel"
        assert "youtube.com/channel/ch1" in result["url"]

    def test_empty_items_fallback(self):
        mock_resp = _mock_httpx_get({"items": []})
        with patch("app.pipelines.yt_ingest.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(
                return_value=MagicMock(get=MagicMock(return_value=mock_resp))
            )
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = fetch_channel_metadata("fake-key", "ch1")

        assert result["title"] == ""
        assert result["handle"] == ""
        assert "youtube.com/channel/ch1" in result["url"]


class TestRunPipeline:
    def test_success(self):
        with (
            patch(
                "app.pipelines.yt_ingest.fetch_video_metadata",
                return_value=VIDEO_API_RESPONSE["items"][0],
            ),
            patch(
                "app.pipelines.yt_ingest.fetch_channel_metadata",
                return_value={
                    "title": "Test Channel",
                    "handle": "@test",
                    "url": "https://www.youtube.com/channel/ch1",
                },
            ),
            patch(
                "app.pipelines.yt_ingest.fetch_transcript",
                return_value=("Hello world", "en"),
            ),
        ):
            result = run_pipeline("fake-key", "v1")

        assert result["channel"]["channel_id"] == "ch1"
        assert result["video"]["video_id"] == "v1"
        assert result["video"]["view_count"] == 12345
        assert result["transcript"]["content"] == "Hello world"

    def test_missing_channel_id(self):
        bad_video = {
            "snippet": {"channelId": "", "title": "No Channel"},
            "statistics": {},
        }
        with patch(
            "app.pipelines.yt_ingest.fetch_video_metadata", return_value=bad_video
        ):
            with pytest.raises(ValueError, match="no channelId"):
                run_pipeline("fake-key", "v1")
