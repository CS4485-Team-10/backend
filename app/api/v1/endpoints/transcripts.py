from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select, func

from app.core.database import get_session
from app.models.transcript import Transcript

router = APIRouter()


@router.get("/transcripts")
def list_transcripts(
    video_id: str = Query(..., description="Video ID (required)"),
    language: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    statement = select(Transcript).where(Transcript.video_id == video_id)
    count_stmt = (
        select(func.count())
        .select_from(Transcript)
        .where(Transcript.video_id == video_id)
    )

    if language:
        statement = statement.where(Transcript.language == language)
        count_stmt = count_stmt.where(Transcript.language == language)

    total = session.exec(count_stmt).one()
    transcripts = session.exec(statement.offset(skip).limit(limit)).all()
    return {"ok": True, "data": transcripts, "count": total}
