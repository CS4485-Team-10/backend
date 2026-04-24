"""
Sentiment Shift endpoint for the Trend Analytics page.

Reads the `sentiment_label` + `sentiment_score` columns that the DS team's
`pipelines/misinfo_checker.py::_run_claim_sentiment` writes into `claims`.
Uses the Supabase REST client — same pattern as alerts.py, creators.py, and
misinfo_checker.py itself — so no direct-postgres DATABASE_URL is required.

Self-contained: brings its own schemas (app.schemas.sentiment) and does not
modify any existing shared module.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from supabase import create_client

from app.core.config import settings
from app.schemas.sentiment import (
    SentimentBucket,
    SentimentShiftResponse,
    SentimentTotals,
)

router = APIRouter()


# range key -> (window_days, bucket_size_days).
# Chosen so every range yields a reasonable number of points:
#   7d  -> 7 daily buckets
#   30d -> 30 daily buckets
#   6m  -> ~26 weekly buckets
#   1y  -> ~27 bi-weekly buckets
_RANGE_CONFIG: dict[str, tuple[int, int]] = {
    "7d": (7, 1),
    "30d": (30, 1),
    "6m": (180, 7),
    "1y": (365, 14),
}


def _normalize_sentiment_label(raw: Optional[str]) -> Optional[str]:
    """Map misinfo_checker's labels (POSITIVE / NEGATIVE / NEUTRAL) to
    canonical lowercase keys. Tolerant of stray casing / variants."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text.startswith("pos"):
        return "positive"
    if text.startswith("neg"):
        return "negative"
    if text.startswith("neu"):
        return "neutral"
    return None


def _parse_created_at(raw: Optional[str]) -> Optional[datetime]:
    """Supabase returns timestamps as ISO strings; normalize to UTC datetime."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@router.get("/overview/sentiment-shift", response_model=SentimentShiftResponse)
def sentiment_shift(
    range_: str = Query(
        "30d",
        alias="range",
        description="Time window for the shift series: 7d | 30d | 6m | 1y",
    ),
):
    if range_ not in _RANGE_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid range '{range_}'. Use one of: {list(_RANGE_CONFIG)}",
        )

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    window_days, bucket_size_days = _RANGE_CONFIG[range_]
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)

    sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    # Pull only the columns we need, filtered to the window and to rows the DS
    # pipeline has actually scored.
    query = (
        sb.table("claims")
        .select("created_at, sentiment_label, sentiment_score")
        .gte("created_at", window_start.isoformat())
    )
    # `.not_.is_("col", "null")` is the PostgREST idiom for IS NOT NULL.
    # Fall back gracefully if the client's API differs — we also re-filter
    # the rows client-side below.
    try:
        query = query.not_.is_("sentiment_label", "null")
    except AttributeError:
        pass

    rows = query.execute().data or []

    # Pre-seed every bucket so the frontend gets a continuous series
    # (no gaps on days with zero claims).
    num_buckets = max(1, (window_days + bucket_size_days - 1) // bucket_size_days)
    bucket_counts: dict[str, dict[str, int]] = {}
    bucket_score_sum: dict[str, float] = defaultdict(float)
    bucket_score_n: dict[str, int] = defaultdict(int)

    for i in range(num_buckets):
        bucket_start = window_start + timedelta(days=i * bucket_size_days)
        key = bucket_start.strftime("%Y-%m-%d")
        bucket_counts[key] = {"positive": 0, "neutral": 0, "negative": 0}

    ordered_keys = sorted(bucket_counts.keys())

    totals = {"positive": 0, "neutral": 0, "negative": 0}
    total_score_sum = 0.0
    total_score_n = 0

    for row in rows:
        label = _normalize_sentiment_label(row.get("sentiment_label"))
        if label is None:
            continue
        created = _parse_created_at(row.get("created_at"))
        if created is None:
            continue
        days_in = (created - window_start).days
        if days_in < 0:
            continue
        bucket_index = min(days_in // bucket_size_days, num_buckets - 1)
        bucket_start = window_start + timedelta(days=bucket_index * bucket_size_days)
        key = bucket_start.strftime("%Y-%m-%d")
        bucket_counts[key][label] += 1
        totals[label] += 1
        raw_score = row.get("sentiment_score")
        if raw_score is not None:
            try:
                score_val = float(raw_score)
            except (TypeError, ValueError):
                continue
            bucket_score_sum[key] += score_val
            bucket_score_n[key] += 1
            total_score_sum += score_val
            total_score_n += 1

    buckets: list[SentimentBucket] = []
    for key in ordered_keys:
        counts = bucket_counts[key]
        total = counts["positive"] + counts["neutral"] + counts["negative"]
        avg = (
            round(bucket_score_sum[key] / bucket_score_n[key], 4)
            if bucket_score_n[key] > 0
            else None
        )
        buckets.append(
            SentimentBucket(
                date=key,
                positive=counts["positive"],
                neutral=counts["neutral"],
                negative=counts["negative"],
                total=total,
                avg_score=avg,
            )
        )

    grand_total = totals["positive"] + totals["neutral"] + totals["negative"]
    overall_avg = (
        round(total_score_sum / total_score_n, 4) if total_score_n > 0 else None
    )

    return SentimentShiftResponse(
        range=range_,
        bucket_size_days=bucket_size_days,
        window_days=window_days,
        totals=SentimentTotals(
            positive=totals["positive"],
            neutral=totals["neutral"],
            negative=totals["negative"],
            total=grand_total,
            avg_score=overall_avg,
        ),
        buckets=buckets,
    )
