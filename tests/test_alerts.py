"""Tests for GET /api/v1/alerts."""

from unittest.mock import patch

from tests.conftest import _build_supabase_mock

NARRATIVES = [
    {
        "narrative_id": "n1",
        "narrative_label": "Ozempic Weight Loss",
        "narrative_description": "Claims about rapid weight loss using Ozempic.",
        "narrative_risk_score": 8.0,
        "created_at": "2026-01-01T00:00:00",
    },
    {
        "narrative_id": "n2",
        "narrative_label": "Vitamin D Cures All",
        "narrative_description": "Misleading claims about vitamin D.",
        "narrative_risk_score": 5.0,
        "created_at": "2026-02-01T00:00:00",
    },
    {
        "narrative_id": "n3",
        "narrative_label": "Safe Supplement",
        "narrative_description": "Low-risk supplement discussion.",
        "narrative_risk_score": 2.0,
        "created_at": "2026-03-01T00:00:00",
    },
]

CLAIM_NARRATIVES = [
    {"claim_id": "cl1", "narrative_id": "n1"},
    {"claim_id": "cl2", "narrative_id": "n2"},
]

CLAIMS = [
    {"claim_id": "cl1", "video_id": "v1", "created_at": "2026-03-01T00:00:00"},
    {"claim_id": "cl2", "video_id": "v2", "created_at": "2026-03-15T00:00:00"},
]

VIDEOS = [
    {"video_id": "v1", "view_count": 50_000},
    {"video_id": "v2", "view_count": 10_000},
]

TABLE_RESPONSES = {
    "narratives": NARRATIVES,
    "claim_narratives": CLAIM_NARRATIVES,
    "claims": CLAIMS,
    "videos": VIDEOS,
}


def _patched_get(client, params=None):
    sb = _build_supabase_mock(TABLE_RESPONSES)
    with patch("app.api.v1.endpoints.alerts.create_client", return_value=sb):
        return client.get("/api/v1/alerts", params=params)


class TestListAlerts:
    def test_success(self, client, supabase_env):
        resp = _patched_get(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["count"] == 3

    def test_sorted_by_risk_desc(self, client, supabase_env):
        resp = _patched_get(client)
        data = resp.json()["data"]
        scores = [r["risk_score"] for r in data]
        assert scores == sorted(scores, reverse=True)

    def test_summary_counts(self, client, supabase_env):
        resp = _patched_get(client)
        body = resp.json()
        assert body["high_count"] == 1
        assert body["medium_count"] == 1
        assert body["low_count"] == 1

    def test_filter_high(self, client, supabase_env):
        resp = _patched_get(client, params={"risk_level": "High"})
        body = resp.json()
        assert body["count"] == 1
        assert body["data"][0]["risk_level"] == "High"
        assert body["high_count"] == 1
        assert body["medium_count"] == 1
        assert body["low_count"] == 1

    def test_filter_medium(self, client, supabase_env):
        resp = _patched_get(client, params={"risk_level": "Medium"})
        body = resp.json()
        assert body["count"] == 1
        assert body["data"][0]["risk_level"] == "Medium"

    def test_filter_low(self, client, supabase_env):
        resp = _patched_get(client, params={"risk_level": "Low"})
        body = resp.json()
        assert body["count"] == 1
        assert body["data"][0]["risk_level"] == "Low"

    def test_filter_returns_no_match(self, client, supabase_env):
        sb = _build_supabase_mock(
            {
                "narratives": [NARRATIVES[2]],
                "claim_narratives": [],
                "claims": [],
                "videos": [],
            }
        )
        with patch("app.api.v1.endpoints.alerts.create_client", return_value=sb):
            resp = client.get("/api/v1/alerts", params={"risk_level": "High"})
        assert resp.json()["count"] == 0

    def test_empty_data(self, client, supabase_env):
        sb = _build_supabase_mock(
            {
                "narratives": [],
                "claim_narratives": [],
                "claims": [],
                "videos": [],
            }
        )
        with patch("app.api.v1.endpoints.alerts.create_client", return_value=sb):
            resp = client.get("/api/v1/alerts")
        body = resp.json()
        assert body["count"] == 0
        assert body["high_count"] == 0

    def test_missing_supabase(self, client, no_supabase_env):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 503

    def test_views_formatted(self, client, supabase_env):
        resp = _patched_get(client)
        data = resp.json()["data"]
        high_alert = next(r for r in data if r["id"] == "n1")
        assert high_alert["total_views"] == "50K"

    def test_videos_analyzed_count(self, client, supabase_env):
        resp = _patched_get(client)
        data = resp.json()["data"]
        high_alert = next(r for r in data if r["id"] == "n1")
        assert high_alert["videos_analyzed"] == 1
