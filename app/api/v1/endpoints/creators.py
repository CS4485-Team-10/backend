from __future__ import annotations

from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from supabase import create_client

from app.core.config import settings

router = APIRouter()


def _fmt_reach(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


@router.get("/creators/risk")
def list_creator_risk(
    narrative: Optional[str] = Query(None, description="Filter by narrative label"),
    sort: str = Query("risk_desc", description="Sort order: risk_desc | risk_asc | reach_desc"),
):
    """Creator Risk Monitor: per-channel risk data from claims + videos."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    videos = sb.table("videos").select("video_id, channel_id, channel_title, view_count").execute().data or []
    claims = sb.table("claims").select("claim_id, video_id, fact_check_status, llm_confidence").execute().data or []
    claim_narratives = sb.table("claim_narratives").select("claim_id, narrative_id").execute().data or []
    narratives = sb.table("narratives").select("narrative_id, narrative_label").execute().data or []

    # Build narrative lookup: claim_id -> set of narrative labels
    narr_by_id: dict[str, str] = {n["narrative_id"]: n["narrative_label"] for n in narratives}
    claim_narr_labels: dict[str, set[str]] = defaultdict(set)
    for cn in claim_narratives:
        label = narr_by_id.get(cn["narrative_id"])
        if label:
            claim_narr_labels[cn["claim_id"]].add(label)

    # Map video_id -> channel_id, aggregate channel reach
    vid_to_channel: dict[str, str] = {}
    channel_handle: dict[str, str] = {}
    channel_reach: dict[str, int] = defaultdict(int)
    for v in videos:
        cid = v.get("channel_id") or ""
        vid = v.get("video_id") or ""
        if cid and vid:
            vid_to_channel[vid] = cid
            channel_reach[cid] += int(v.get("view_count") or 0)
            if cid not in channel_handle:
                channel_handle[cid] = v.get("channel_title") or cid

    # Aggregate claims per channel
    channel_flagged: dict[str, int] = defaultdict(int)
    channel_confidence_sum: dict[str, float] = defaultdict(float)
    channel_claim_count: dict[str, int] = defaultdict(int)
    channel_narratives_set: dict[str, set[str]] = defaultdict(set)

    for cl in claims:
        vid = cl.get("video_id") or ""
        cid = vid_to_channel.get(vid)
        if not cid:
            continue

        channel_claim_count[cid] += 1
        status = (cl.get("fact_check_status") or "").lower()
        if status in ("disputed", "flagged", "false", "unverified"):
            channel_flagged[cid] += 1

        conf = cl.get("llm_confidence")
        if conf is not None:
            channel_confidence_sum[cid] += float(conf)

        for label in claim_narr_labels.get(cl["claim_id"], set()):
            channel_narratives_set[cid].add(label)

    # Build results per unique channel
    results = []
    seen_channels: set[str] = set()
    for v in videos:
        cid = v.get("channel_id") or ""
        if not cid or cid in seen_channels:
            continue
        seen_channels.add(cid)

        if narrative:
            narr_titles = channel_narratives_set.get(cid, set())
            if not any(narrative.lower() in t.lower() for t in narr_titles):
                continue

        n_claims = channel_claim_count.get(cid, 0)
        if n_claims > 0:
            avg_conf = channel_confidence_sum[cid] / n_claims
            risk_score = round(avg_conf * 10, 1)
        else:
            risk_score = 0.0

        results.append({
            "handle": channel_handle.get(cid, cid),
            "risk_score": risk_score,
            "flagged_claims": channel_flagged.get(cid, 0),
            "reach": _fmt_reach(channel_reach.get(cid, 0)),
        })

    # Keep numeric reach for sorting; look up by handle
    handle_to_reach: dict[str, int] = {}
    for cid, handle in channel_handle.items():
        handle_to_reach[handle] = channel_reach.get(cid, 0)

    if sort == "risk_asc":
        results.sort(key=lambda r: r["risk_score"])
    elif sort == "reach_desc":
        results.sort(key=lambda r: handle_to_reach.get(r["handle"], 0), reverse=True)
    else:
        results.sort(key=lambda r: -r["risk_score"])

    return {"ok": True, "data": results, "count": len(results)}
