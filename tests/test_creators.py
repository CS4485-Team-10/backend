"""Tests for GET /api/v1/creators/risk."""

from unittest.mock import patch

from tests.conftest import _build_supabase_mock

VIDEOS = [
    {
        "video_id": "v1",
        "channel_id": "ch1",
        "channel_title": "Alpha Channel",
        "view_count": 50_000,
    },
    {
        "video_id": "v2",
        "channel_id": "ch1",
        "channel_title": "Alpha Channel",
        "view_count": 30_000,
    },
    {
        "video_id": "v3",
        "channel_id": "ch2",
        "channel_title": "Beta Channel",
        "view_count": 1_000_000,
    },
]

CLAIMS = [
    {
        "claim_id": "cl1",
        "video_id": "v1",
        "fact_check_status": "disputed",
        "llm_confidence": 0.9,
    },
    {
        "claim_id": "cl2",
        "video_id": "v1",
        "fact_check_status": "verified",
        "llm_confidence": 0.3,
    },
    {
        "claim_id": "cl3",
        "video_id": "v3",
        "fact_check_status": "flagged",
        "llm_confidence": 0.7,
    },
]

CLAIM_NARRATIVES = [
    {"claim_id": "cl1", "narrative_id": "n1"},
    {"claim_id": "cl3", "narrative_id": "n2"},
]

NARRATIVES = [
    {"narrative_id": "n1", "narrative_label": "Ozempic Weight Loss"},
    {"narrative_id": "n2", "narrative_label": "Vaccine Controversy"},
]

TABLE_RESPONSES = {
    "videos": VIDEOS,
    "claims": CLAIMS,
    "claim_narratives": CLAIM_NARRATIVES,
    "narratives": NARRATIVES,
}


def _patched_get(client, params=None):
    sb = _build_supabase_mock(TABLE_RESPONSES)
    with patch("app.api.v1.endpoints.creators.create_client", return_value=sb):
        return client.get("/api/v1/creators/risk", params=params)


class TestListCreatorRisk:
    def test_success(self, client, supabase_env):
        resp = _patched_get(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["count"] == 2

    def test_default_sort_risk_desc(self, client, supabase_env):
        resp = _patched_get(client)
        data = resp.json()["data"]
        scores = [r["risk_score"] for r in data]
        assert scores == sorted(scores, reverse=True)

    def test_sort_risk_asc(self, client, supabase_env):
        resp = _patched_get(client, params={"sort": "risk_asc"})
        data = resp.json()["data"]
        scores = [r["risk_score"] for r in data]
        assert scores == sorted(scores)

    def test_sort_reach_desc(self, client, supabase_env):
        resp = _patched_get(client, params={"sort": "reach_desc"})
        data = resp.json()["data"]
        assert data[0]["handle"] == "Beta Channel"

    def test_narrative_filter(self, client, supabase_env):
        resp = _patched_get(client, params={"narrative": "Ozempic"})
        body = resp.json()
        assert body["count"] == 1
        assert body["data"][0]["handle"] == "Alpha Channel"

    def test_narrative_filter_no_match(self, client, supabase_env):
        resp = _patched_get(client, params={"narrative": "Nonexistent"})
        assert resp.json()["count"] == 0

    def test_empty_data(self, client, supabase_env):
        sb = _build_supabase_mock(
            {"videos": [], "claims": [], "claim_narratives": [], "narratives": []}
        )
        with patch("app.api.v1.endpoints.creators.create_client", return_value=sb):
            resp = client.get("/api/v1/creators/risk")
        assert resp.json()["count"] == 0

    def test_missing_supabase(self, client, no_supabase_env):
        resp = client.get("/api/v1/creators/risk")
        assert resp.status_code == 503

    def test_flagged_claims_counted(self, client, supabase_env):
        resp = _patched_get(client)
        data = resp.json()["data"]
        alpha = next(r for r in data if r["handle"] == "Alpha Channel")
        assert alpha["flagged_claims"] == 1
