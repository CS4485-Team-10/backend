from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, col, func, select

from app.core.database import get_session
from app.models.claim import Claim
from app.models.video import Video
from app.schemas.claim import ClaimListResponse, ClaimResponse, ClaimStatsResponse

router = APIRouter()


def _format_views(total: int | None) -> str:
    if total is None or total == 0:
        return "0"
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    if total >= 1_000:
        return f"{total / 1_000:.1f}K"
    return str(total)


def _claim_status_display(claim: Claim) -> str:
    if claim.fact_check_status:
        return claim.fact_check_status
    if claim.risk_level:
        return claim.risk_level.title()
    return "Under Review"


def _claim_confidence_display(claim: Claim) -> str:
    if claim.fact_check_confidence:
        return claim.fact_check_confidence
    if claim.llm_confidence is not None:
        return f"{float(claim.llm_confidence):.2f}"
    return "—"


def _build_claim_response(
    claim: Claim,
    channel_title: str | None,
    view_count: int | None,
) -> ClaimResponse:
    return ClaimResponse(
        claim_id=str(claim.claim_id)[:8].upper(),
        text=claim.claim_text,
        source=channel_title or "Unknown",
        status=_claim_status_display(claim),
        confidence=_claim_confidence_display(claim),
        views=_format_views(view_count),
        date=claim.created_at.strftime("%Y-%m-%d"),
    )


@router.get("/claims", response_model=ClaimListResponse)
def list_claims(
    search: str | None = Query(None),
    status: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    statement = select(Claim, Video.channel_title, Video.view_count).join(
        Video, Claim.video_id == Video.video_id
    )

    count_stmt = select(func.count()).select_from(Claim)

    if search:
        pattern = f"%{search}%"
        statement = statement.where(
            col(Claim.claim_text).ilike(pattern)
            | col(Video.channel_title).ilike(pattern)
        )
        count_stmt = (
            select(func.count())
            .select_from(Claim)
            .join(Video, Claim.video_id == Video.video_id)
            .where(
                col(Claim.claim_text).ilike(pattern)
                | col(Video.channel_title).ilike(pattern)
            )
        )

    if status:
        statement = statement.where(col(Claim.fact_check_status).ilike(status))
        count_stmt = (
            select(func.count())
            .select_from(Claim)
            .where(col(Claim.fact_check_status).ilike(status))
        )

    total = session.exec(count_stmt).one()

    statement = statement.order_by(Claim.created_at.desc()).offset(skip).limit(limit)
    rows = session.exec(statement).all()

    data = [
        _build_claim_response(claim, channel_title, view_count)
        for claim, channel_title, view_count in rows
    ]

    verified = session.exec(
        select(func.count())
        .select_from(Claim)
        .where(col(Claim.fact_check_status).ilike("verified"))
    ).one()
    disputed = session.exec(
        select(func.count())
        .select_from(Claim)
        .where(col(Claim.fact_check_status).ilike("disputed"))
    ).one()
    under_review = total - verified - disputed

    stats = ClaimStatsResponse(
        total=total,
        verified=verified,
        disputed=disputed,
        under_review=under_review,
    )

    return ClaimListResponse(data=data, stats=stats, count=total)


@router.get("/claims/{claim_id}")
def get_claim(
    claim_id: str,
    session: Session = Depends(get_session),
):
    row = session.exec(
        select(Claim, Video.channel_title, Video.view_count)
        .join(Video, Claim.video_id == Video.video_id)
        .where(Claim.claim_id == claim_id)
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim, channel_title, view_count = row
    return {
        "ok": True,
        "data": _build_claim_response(claim, channel_title, view_count),
    }
