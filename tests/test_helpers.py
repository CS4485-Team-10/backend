"""Unit tests for pure helper functions — no mocking required."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.api.v1.endpoints.alerts import _fmt_views, _parse_risk, _time_window
from app.api.v1.endpoints.creators import _fmt_reach
from app.pipelines.yt_ingest import clean_transcript


# ── _fmt_views / _fmt_reach (identical logic) ──────────────────────────


class TestFmtViews:
    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, "0"),
            (999, "999"),
            (1_000, "1.0K"),
            (1_500, "1.5K"),
            (9_999, "10.0K"),
            (10_000, "10K"),
            (46_000, "46K"),
            (999_999, "1000K"),
            (1_000_000, "1.0M"),
            (1_500_000, "1.5M"),
            (15_300_000, "15.3M"),
        ],
    )
    def test_format(self, n, expected):
        assert _fmt_views(n) == expected


class TestFmtReach:
    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, "0"),
            (500, "500"),
            (2_600, "2.6K"),
            (12_000, "12K"),
            (1_400_000, "1.4M"),
        ],
    )
    def test_format(self, n, expected):
        assert _fmt_reach(n) == expected


# ── _parse_risk ─────────────────────────────────────────────────────────


class TestParseRisk:
    def test_none(self):
        assert _parse_risk(None) == ("Low", 0.0)

    @pytest.mark.parametrize(
        "raw, level, score",
        [
            (8.0, "High", 8.0),
            (7.0, "High", 7.0),
            (5.0, "Medium", 5.0),
            (4.0, "Medium", 4.0),
            (2.0, "Low", 2.0),
            (0.0, "Low", 0.0),
        ],
    )
    def test_numeric(self, raw, level, score):
        assert _parse_risk(raw) == (level, score)

    @pytest.mark.parametrize(
        "raw, level, score",
        [
            ("high", "High", 8.0),
            ("High", "High", 8.0),
            ("medium", "Medium", 5.0),
            ("low", "Low", 2.0),
        ],
    )
    def test_string_labels(self, raw, level, score):
        assert _parse_risk(raw) == (level, score)

    def test_string_numeric(self):
        assert _parse_risk("9.5") == ("High", 9.5)

    def test_unknown_string(self):
        assert _parse_risk("unknown") == ("Low", 0.0)

    def test_integer_input(self):
        assert _parse_risk(7) == ("High", 7.0)


# ── _time_window ────────────────────────────────────────────────────────


class TestTimeWindow:
    def test_none(self):
        assert _time_window(None) == "All time"

    def test_empty_string(self):
        assert _time_window("") == "All time"

    def test_last_30_days(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        assert _time_window(ts) == "Last 30 days"

    def test_last_60_days(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        assert _time_window(ts) == "Last 60 days"

    def test_last_90_days(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=75)).isoformat()
        assert _time_window(ts) == "Last 90 days"

    def test_last_120_days(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        assert _time_window(ts) == "Last 120 days"

    def test_beyond_120_days(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        assert _time_window(ts) == "Last 200 days"

    def test_invalid_string(self):
        assert _time_window("not-a-date") == "All time"


# ── clean_transcript ────────────────────────────────────────────────────


class TestCleanTranscript:
    def test_basic_join(self):
        data = [{"text": "Hello"}, {"text": "world"}]
        result = clean_transcript(data)
        assert result == "Hello world"

    def test_removes_brackets(self):
        data = [{"text": "Hello [Music] world"}]
        result = clean_transcript(data)
        assert "Music" not in result

    def test_removes_parentheses(self):
        data = [{"text": "Hello (applause) world"}]
        result = clean_transcript(data)
        assert "applause" not in result

    def test_removes_filler_words(self):
        data = [{"text": "So um the thing is uh important"}]
        result = clean_transcript(data)
        assert "um" not in result.lower().split()
        assert "uh" not in result.lower().split()

    def test_capitalizes_first_letter(self):
        data = [{"text": "hello world"}]
        result = clean_transcript(data)
        assert result[0] == "H"

    def test_empty_input(self):
        assert clean_transcript([]) == ""

    def test_collapses_whitespace(self):
        data = [{"text": "lots   of    spaces"}]
        result = clean_transcript(data)
        assert "  " not in result
