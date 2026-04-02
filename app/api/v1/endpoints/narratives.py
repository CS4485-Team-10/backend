from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func, col

from app.core.database import get_session
from app.models.narrative import Narrative
from app.models.claim import Claim
from app.models.claim_narrative import ClaimNarrative
from app.models.video import Video
from app.schemas.narrative import NarrativeListResponse, NarrativeResponse

router = APIRouter()


def _format_views(total: Optional[int]) -> str:
    if total is None or total == 0:
        return "0"
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    if total >= 1_000:
        return f"{total / 1_000:.1f}K"
    return str(total)


def _time_window(earliest: Optional[datetime]) -> str:
    if earliest is None:
        return "No data"
    days = (datetime.utcnow() - earliest).days
    if days <= 30:
        return "Last 30 days"
    if days <= 60:
        return "Last 60 days"
    if days <= 90:
        return "Last 90 days"
    if days <= 120:
        return "Last 120 days"
    return f"Last {days} days"


def _build_narrative_response(
    narrative: Narrative,
    videos_analyzed: int,
    raw_views: Optional[int],
    earliest_claim: Optional[datetime],
) -> NarrativeResponse:
    return NarrativeResponse(
        id=str(narrative.narrative_id),
        title=narrative.narrative_label,
        description=narrative.narrative_description,
        # TODO: separate detail field once narratives.narrative_detail column exists
        detail=narrative.narrative_description,
        # TODO: populate once narratives.category column exists
        category="Uncategorized",
        # TODO: compute from risk scoring once narratives.risk_level exists
        risk_level="Medium",
        # TODO: compute from risk scoring once narratives.risk_score exists
        risk_score=5.0,
        videos_analyzed=videos_analyzed,
        total_views=_format_views(raw_views),
        time_window=_time_window(earliest_claim),
        # TODO: populate once narratives.primary_link column exists
        primary_link=None,
    )


@router.get("/narratives", response_model=NarrativeListResponse)
def list_narratives(
    search: Optional[str] = Query(None),
    sort_by: Optional[str] = Query("trending"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    # Base query: narrative + aggregated stats
    statement = (
        select(
            Narrative,
            func.count(func.distinct(Claim.video_id)).label("videos_analyzed"),
            func.sum(Video.view_count).label("raw_views"),
            func.min(Claim.created_at).label("earliest_claim"),
        )
        .outerjoin(ClaimNarrative, Narrative.narrative_id == ClaimNarrative.narrative_id)
        .outerjoin(Claim, ClaimNarrative.claim_id == Claim.claim_id)
        .outerjoin(Video, Claim.video_id == Video.video_id)
        .group_by(Narrative.narrative_id)
    )

    if search:
        pattern = f"%{search}%"
        statement = statement.where(
            col(Narrative.narrative_label).ilike(pattern)
            | col(Narrative.narrative_description).ilike(pattern)
        )

    # Sorting
    if sort_by == "views":
        statement = statement.order_by(func.sum(Video.view_count).desc().nulls_last())
    elif sort_by == "risk":
        # TODO: sort by real risk_score once column exists; for now newest first
        statement = statement.order_by(Narrative.created_at.desc())
    else:
        # trending = newest first
        statement = statement.order_by(Narrative.created_at.desc())

    # Count before pagination
    count_stmt = select(func.count()).select_from(Narrative)
    if search:
        pattern = f"%{search}%"
        count_stmt = count_stmt.where(
            col(Narrative.narrative_label).ilike(pattern)
            | col(Narrative.narrative_description).ilike(pattern)
        )
    total = session.exec(count_stmt).one()

    rows = session.exec(statement.offset(skip).limit(limit)).all()

    data = [
        _build_narrative_response(narrative, videos_analyzed or 0, raw_views, earliest_claim)
        for narrative, videos_analyzed, raw_views, earliest_claim in rows
    ]

    return NarrativeListResponse(data=data, count=total)


@router.get("/narratives/{narrative_id}")
def get_narrative(
    narrative_id: str,
    session: Session = Depends(get_session),
):
    row = session.exec(
        select(
            Narrative,
            func.count(func.distinct(Claim.video_id)).label("videos_analyzed"),
            func.sum(Video.view_count).label("raw_views"),
            func.min(Claim.created_at).label("earliest_claim"),
        )
        .outerjoin(ClaimNarrative, Narrative.narrative_id == ClaimNarrative.narrative_id)
        .outerjoin(Claim, ClaimNarrative.claim_id == Claim.claim_id)
        .outerjoin(Video, Claim.video_id == Video.video_id)
        .where(Narrative.narrative_id == narrative_id)
        .group_by(Narrative.narrative_id)
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Narrative not found")

    narrative, videos_analyzed, raw_views, earliest_claim = row
    return {
        "ok": True,
        "data": _build_narrative_response(narrative, videos_analyzed or 0, raw_views, earliest_claim),
    }
