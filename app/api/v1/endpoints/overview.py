from fastapi import APIRouter, Depends
from sqlmodel import Session, select, func

from app.core.database import get_session
from app.models.video import Video
from app.models.insight import Insight
from app.schemas.overview import OverviewStatsResponse

router = APIRouter()


@router.get("/overview/stats", response_model=OverviewStatsResponse)
def overview_stats(session: Session = Depends(get_session)):
    total_videos = session.exec(select(func.count()).select_from(Video)).one()

    insights = session.exec(select(Insight.narratives, Insight.claims)).all()

    unique_narratives: set[str] = set()
    total_claims = 0
    for narratives, claims in insights:
        if isinstance(narratives, list):
            for n in narratives:
                if isinstance(n, str):
                    unique_narratives.add(n)
                elif isinstance(n, dict) and "text" in n:
                    unique_narratives.add(n["text"])
        if isinstance(claims, list):
            total_claims += len(claims)

    return OverviewStatsResponse(
        total_videos_scoped=total_videos,
        active_narratives=len(unique_narratives),
        total_claims=total_claims,
        verified_claims=None,
        high_risk_alerts=None,
    )
