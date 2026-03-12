from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func

from app.core.database import get_session
from app.models.insight import Insight

router = APIRouter()


@router.get("/insights")
def list_insights(
    video_id: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    statement = select(Insight)
    count_stmt = select(func.count()).select_from(Insight)

    if video_id:
        statement = statement.where(Insight.video_id == video_id)
        count_stmt = count_stmt.where(Insight.video_id == video_id)
    if model:
        statement = statement.where(Insight.model == model)
        count_stmt = count_stmt.where(Insight.model == model)

    total = session.exec(count_stmt).one()
    insights = session.exec(statement.offset(skip).limit(limit)).all()
    return {"ok": True, "data": insights, "count": total}


@router.get("/insights/{insight_id}")
def get_insight(insight_id: int, session: Session = Depends(get_session)):
    insight = session.get(Insight, insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return {"ok": True, "data": insight}
