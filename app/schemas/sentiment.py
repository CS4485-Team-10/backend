"""
Pydantic schemas for the Sentiment Shift endpoint (Trend Analytics page).

Kept in its own module so this feature doesn't depend on edits to
schemas/overview.py or other shared schema files.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SentimentBucket(BaseModel):
    date: str
    positive: int
    neutral: int
    negative: int
    total: int
    avg_score: Optional[float] = None


class SentimentTotals(BaseModel):
    positive: int
    neutral: int
    negative: int
    total: int
    avg_score: Optional[float] = None


class SentimentShiftResponse(BaseModel):
    ok: bool = True
    range: str
    bucket_size_days: int
    window_days: int
    totals: SentimentTotals
    buckets: list[SentimentBucket]
