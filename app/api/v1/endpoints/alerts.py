from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from supabase import create_client

from app.core.config import settings

router = APIRouter()


# --- Helper: format a raw view count into a readable string ---
# 1500000 -> "1.5M", 46000 -> "46K", 800 -> "800"
def _fmt_views(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# --- Helper: convert the DB's narrative_risk value into (level, score) ---
# The DB stores this as a string like "high"/"medium"/"low" OR a number 0-10.
# The frontend needs both a label ("High") and a numeric score (8.0).
def _parse_risk(raw: str | float | None) -> tuple[str, float]:
    if raw is None:
        return ("Low", 0.0)
    # If it's already a number, map it to a level based on thresholds
    if isinstance(raw, (int, float)):
        score = float(raw)
        if score >= 7:
            return ("High", score)
        if score >= 4:
            return ("Medium", score)
        return ("Low", score)
    # If it's a string like "high", assign a default numeric score
    text = str(raw).strip().lower()
    if text == "high":
        return ("High", 8.0)
    if text == "medium":
        return ("Medium", 5.0)
    if text == "low":
        return ("Low", 2.0)
    # Last resort: try to parse the string as a number
    try:
        return _parse_risk(float(text))
    except ValueError:
        return ("Low", 0.0)


# --- Helper: figure out a human-readable time window from a timestamp ---
# Compares the earliest claim date to "now" and returns e.g. "Last 30 days"
def _time_window(earliest_iso: str | None) -> str:
    if not earliest_iso:
        return "All time"
    try:
        earliest = datetime.fromisoformat(earliest_iso.replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - earliest).days
        if days <= 30:
            return "Last 30 days"
        if days <= 60:
            return "Last 60 days"
        if days <= 90:
            return "Last 90 days"
        if days <= 120:
            return "Last 120 days"
        return f"Last {days} days"
    except (ValueError, TypeError):
        return "All time"


# ============================================================
# THE ACTUAL ENDPOINT — this is what the frontend calls
# URL: GET /api/v1/alerts
# Optional query param: ?risk_level=High (or Medium or Low)
# ============================================================
@router.get("/alerts")
def list_alerts(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: High | Medium | Low"),
):
    # --- Step 0: Make sure Supabase is configured ---
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    # --- Step 1: Pull data from 4 tables ---
    # narratives:       the main list (title, description, risk level)
    # claim_narratives:  join table linking claims <-> narratives
    # claims:           individual health claims extracted from videos
    # videos:           YouTube videos with view counts
    narratives = sb.table("narratives").select("narrative_id, narrative_label, narrative_description, narrative_risk_score, created_at").execute().data or []
    claim_narrs = sb.table("claim_narratives").select("claim_id, narrative_id").execute().data or []
    claims = sb.table("claims").select("claim_id, video_id, created_at").execute().data or []
    videos = sb.table("videos").select("video_id, view_count").execute().data or []

    # --- Step 2: Build quick-lookup dicts so we don't loop repeatedly ---
    # video_id -> view count (e.g. "abc123" -> 46000)
    view_map: dict[str, int] = {v["video_id"]: int(v.get("view_count") or 0) for v in videos}
    # claim_id -> which video it came from
    claim_video: dict[str, str] = {c["claim_id"]: c.get("video_id", "") for c in claims}
    # claim_id -> when the claim was created
    claim_created: dict[str, str] = {c["claim_id"]: c.get("created_at", "") for c in claims}

    # --- Step 3: Group claims by narrative ---
    # narrative_id -> list of claim_ids linked to it
    narr_claims: dict[str, list[str]] = defaultdict(list)
    for cn in claim_narrs:
        narr_claims[cn["narrative_id"]].append(cn["claim_id"])

    # --- Step 4: Loop through each narrative and build the response ---
    results = []
    # These counters power the summary cards at the top of the page
    high_count = 0
    medium_count = 0
    low_count = 0

    for n in narratives:
        nid = n["narrative_id"]

        # Convert the DB's risk value into "High"/"Medium"/"Low" + a 0-10 score
        level, score = _parse_risk(n.get("narrative_risk_score"))

        # Always count for the summary cards (even if we filter it out below)
        if level == "High":
            high_count += 1
        elif level == "Medium":
            medium_count += 1
        else:
            low_count += 1

        # If the frontend sent ?risk_level=High, skip non-High narratives
        if risk_level and level != risk_level:
            continue

        # Find all videos linked to this narrative (through claims)
        # narrative -> claim_narratives -> claims -> videos
        linked_claims = narr_claims.get(nid, [])
        linked_videos: set[str] = set()
        earliest_claim: str | None = None
        for cid in linked_claims:
            vid = claim_video.get(cid, "")
            if vid:
                linked_videos.add(vid)
            # Track the earliest claim date for the "time_window" field
            ts = claim_created.get(cid, "")
            if ts and (earliest_claim is None or ts < earliest_claim):
                earliest_claim = ts

        # Sum up view counts across all linked videos
        total_views = sum(view_map.get(vid, 0) for vid in linked_videos)

        # Build the object exactly how the frontend expects it
        results.append({
            "id": nid,                                        # unique ID
            "title": n.get("narrative_label") or "",          # e.g. "Ozempic Weight Loss"
            "risk_level": level,                              # "High" / "Medium" / "Low"
            "risk_score": round(score, 1),                    # 0.0 - 10.0
            "videos_analyzed": len(linked_videos),            # count of unique videos
            "total_views": _fmt_views(total_views),           # formatted like "15.3M"
            "description": n.get("narrative_description") or "",
            "time_window": _time_window(earliest_claim),      # e.g. "Last 90 days"
        })

    # --- Step 5: Sort by risk score (highest first) and return ---
    results.sort(key=lambda r: -r["risk_score"])

    return {
        "ok": True,
        "data": results,              # the array of narrative alerts
        "count": len(results),        # how many after filtering
        "high_count": high_count,     # for the "High Risk: X" summary card
        "medium_count": medium_count, # for the "Medium Risk: X" summary card
        "low_count": low_count,       # for the "Low Risk: X" summary card
    }
