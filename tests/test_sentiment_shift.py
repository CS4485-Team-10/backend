"""Tests for GET /overview/sentiment-shift.

Self-contained: mounts only the sentiment router onto a dedicated FastAPI
app so the suite isn't affected by unrelated issues in other endpoint
modules. Builds its own Supabase mock (handles .gte, .not_.is_ chains the
existing conftest helper doesn't cover).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints import sentiment as sentiment_endpoint


def _build_chain_mock(data):
    """A chainable mock: every attribute access returns self, so
    `.select().gte().not_.is_().execute()` all flow to the same object."""
    chain = MagicMock()
    chain.execute.return_value = MagicMock(data=data)

    def _return_self(*_a, **_kw):
        return chain

    for method in ("select", "gte", "lte", "eq", "limit", "order", "range"):
        getattr(chain, method).side_effect = _return_self
    # `.not_.is_("col", "null")` pattern: expose a `not_` attr that is another
    # chain returning itself.
    chain.not_ = chain
    chain.is_.side_effect = _return_self
    return chain


def _build_sb_mock(rows):
    sb = MagicMock()
    sb.table.return_value = _build_chain_mock(rows)
    return sb


@pytest.fixture()
def sentiment_client():
    app = FastAPI()
    app.include_router(sentiment_endpoint.router, prefix="/api/v1")
    return TestClient(app)


@pytest.fixture()
def supabase_env(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.SUPABASE_URL", "https://fake.supabase.co"
    )
    monkeypatch.setattr(
        "app.core.config.settings.SUPABASE_SERVICE_ROLE_KEY", "fake-key"
    )


@pytest.fixture()
def no_supabase_env(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.SUPABASE_URL", "")
    monkeypatch.setattr("app.core.config.settings.SUPABASE_SERVICE_ROLE_KEY", "")


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat()


def _claim(label, score, days_ago):
    return {
        "created_at": _iso(_now() - timedelta(days=days_ago)),
        "sentiment_label": label,
        "sentiment_score": score,
    }


# Mirrors exactly what misinfo_checker._run_claim_sentiment writes:
# sentiment_label is "POSITIVE" | "NEGATIVE" | "NEUTRAL" (uppercase),
# sentiment_score is a gradient in [-1.0, +1.0].
SAMPLE_CLAIMS = [
    _claim("POSITIVE", 0.8542, 1),
    _claim("POSITIVE", 0.2100, 2),
    _claim("NEGATIVE", -0.6732, 3),
    _claim("NEUTRAL", 0.0450, 4),
    _claim("NEUTRAL", -0.0200, 5),
    _claim("positive", 0.3, 6),  # stray lowercase — still counts as positive
    _claim(None, None, 1),  # null label — must be ignored
    _claim("POSITIVE", 0.9, 200),  # outside short windows, inside 1y
]


def _call(client, params=None, rows=None):
    data = SAMPLE_CLAIMS if rows is None else rows
    sb = _build_sb_mock(data)
    with patch(
        "app.api.v1.endpoints.sentiment.create_client", return_value=sb
    ):
        return client.get("/api/v1/overview/sentiment-shift", params=params)


class TestSentimentShiftShape:
    def test_response_keys(self, sentiment_client, supabase_env):
        resp = _call(sentiment_client)
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "ok",
            "range",
            "bucket_size_days",
            "window_days",
            "totals",
            "buckets",
        }
        assert body["ok"] is True
        assert set(body["totals"]) == {
            "positive",
            "neutral",
            "negative",
            "total",
            "avg_score",
        }
        assert body["buckets"], "expected non-empty bucket grid"
        assert set(body["buckets"][0]) == {
            "date",
            "positive",
            "neutral",
            "negative",
            "total",
            "avg_score",
        }


class TestRangeWindows:
    def test_default_is_30d(self, sentiment_client, supabase_env):
        body = _call(sentiment_client).json()
        assert body["range"] == "30d"
        assert body["window_days"] == 30
        assert body["bucket_size_days"] == 1
        assert len(body["buckets"]) == 30

    def test_7d(self, sentiment_client, supabase_env):
        body = _call(sentiment_client, {"range": "7d"}).json()
        assert body["range"] == "7d"
        assert body["window_days"] == 7
        assert body["bucket_size_days"] == 1
        assert len(body["buckets"]) == 7

    def test_6m_weekly(self, sentiment_client, supabase_env):
        body = _call(sentiment_client, {"range": "6m"}).json()
        assert body["window_days"] == 180
        assert body["bucket_size_days"] == 7
        assert len(body["buckets"]) == 26  # ceil(180/7)

    def test_1y_biweekly(self, sentiment_client, supabase_env):
        body = _call(sentiment_client, {"range": "1y"}).json()
        assert body["window_days"] == 365
        assert body["bucket_size_days"] == 14
        assert len(body["buckets"]) == 27  # ceil(365/14)

    def test_bad_range_400(self, sentiment_client, supabase_env):
        resp = _call(sentiment_client, {"range": "bogus"})
        assert resp.status_code == 400
        assert "Invalid range" in resp.json()["detail"]


class TestDSLabelHandling:
    """Confirms the endpoint reads exactly what misinfo_checker writes."""

    def test_uppercase_labels_counted(self, sentiment_client, supabase_env):
        # In 7d window: 3 POSITIVE (incl. stray "positive"), 1 NEGATIVE, 2 NEUTRAL
        body = _call(sentiment_client, {"range": "7d"}).json()
        assert body["totals"]["positive"] == 3
        assert body["totals"]["negative"] == 1
        assert body["totals"]["neutral"] == 2
        assert body["totals"]["total"] == 6

    def test_null_labels_skipped(self, sentiment_client, supabase_env):
        # One row has sentiment_label=None; it must not inflate any bucket.
        body = _call(sentiment_client, {"range": "7d"}).json()
        total = sum(
            b["positive"] + b["neutral"] + b["negative"] for b in body["buckets"]
        )
        assert total == 6  # 7 rows with created_at<=7d but 1 has null label

    def test_outside_window_excluded(self, sentiment_client, supabase_env):
        # The -200d POSITIVE is outside 30d but inside 1y. The mock returns
        # all rows regardless of filter; the endpoint must still reject the
        # too-old row client-side.
        short = _call(sentiment_client, {"range": "30d"}).json()
        longw = _call(sentiment_client, {"range": "1y"}).json()
        assert short["totals"]["total"] == 6
        assert longw["totals"]["total"] == 7

    def test_avg_score_matches_arithmetic_mean(self, sentiment_client, supabase_env):
        # In 7d: scores are 0.8542, 0.21, -0.6732, 0.045, -0.02, 0.3
        body = _call(sentiment_client, {"range": "7d"}).json()
        expected = round(
            (0.8542 + 0.21 + (-0.6732) + 0.045 + (-0.02) + 0.3) / 6, 4
        )
        assert body["totals"]["avg_score"] == pytest.approx(expected, abs=1e-4)


class TestBucketInvariants:
    def test_buckets_sorted_and_unique(self, sentiment_client, supabase_env):
        body = _call(sentiment_client).json()
        dates = [b["date"] for b in body["buckets"]]
        assert dates == sorted(dates)
        assert len(set(dates)) == len(dates)

    def test_bucket_sums_equal_totals(self, sentiment_client, supabase_env):
        body = _call(sentiment_client).json()
        sum_pos = sum(b["positive"] for b in body["buckets"])
        sum_neg = sum(b["negative"] for b in body["buckets"])
        sum_neu = sum(b["neutral"] for b in body["buckets"])
        assert sum_pos == body["totals"]["positive"]
        assert sum_neg == body["totals"]["negative"]
        assert sum_neu == body["totals"]["neutral"]

    def test_empty_data_returns_zero_totals_with_full_bucket_grid(
        self, sentiment_client, supabase_env
    ):
        body = _call(sentiment_client, rows=[]).json()
        assert body["totals"] == {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "total": 0,
            "avg_score": None,
        }
        assert len(body["buckets"]) == 30


class TestConfigGuards:
    def test_missing_supabase_returns_503(self, sentiment_client, no_supabase_env):
        resp = sentiment_client.get("/api/v1/overview/sentiment-shift")
        assert resp.status_code == 503
