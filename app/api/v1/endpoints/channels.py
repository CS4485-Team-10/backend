from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, func, select

from app.core.database import get_session
from app.models.video import Video

router = APIRouter()


@router.get("/channels")
def list_channels(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    rows = session.exec(
        select(
            Video.channel_id,
            Video.channel_title,
            func.count(Video.video_id).label("video_count"),
        )
        .group_by(Video.channel_id, Video.channel_title)
        .offset(skip)
        .limit(limit)
    ).all()

    total = session.exec(select(func.count(func.distinct(Video.channel_id)))).one()

    data = [
        {
            "channel_id": channel_id,
            "title": channel_title or "Unknown",
            "video_count": video_count,
        }
        for channel_id, channel_title, video_count in rows
    ]
    return {"ok": True, "data": data, "count": total}


@router.get("/channels/{channel_id}")
def get_channel(channel_id: str, session: Session = Depends(get_session)):
    row = session.exec(
        select(
            Video.channel_id,
            Video.channel_title,
            func.count(Video.video_id).label("video_count"),
        )
        .where(Video.channel_id == channel_id)
        .group_by(Video.channel_id, Video.channel_title)
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Channel not found")

    channel_id, channel_title, video_count = row
    return {
        "ok": True,
        "data": {
            "channel_id": channel_id,
            "title": channel_title or "Unknown",
            "video_count": video_count,
        },
    }
