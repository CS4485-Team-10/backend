from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, func, select

from app.core.database import get_session
from app.models.channel import Channel
from app.models.video import Video
from app.schemas.video import VideoWithChannel

router = APIRouter()


@router.get("/videos")
def list_videos(
    channel_id: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    statement = select(Video, Channel.title, Channel.handle).join(
        Channel, Video.channel_id == Channel.channel_id
    )
    if channel_id:
        statement = statement.where(Video.channel_id == channel_id)

    count_stmt = select(func.count()).select_from(Video)
    if channel_id:
        count_stmt = count_stmt.where(Video.channel_id == channel_id)
    total = session.exec(count_stmt).one()

    results = session.exec(statement.offset(skip).limit(limit)).all()
    data = [
        VideoWithChannel(
            **video.model_dump(),
            channel_title=ch_title,
            channel_handle=ch_handle,
        )
        for video, ch_title, ch_handle in results
    ]
    return {"ok": True, "data": data, "count": total}


@router.get("/videos/{video_id}")
def get_video(video_id: str, session: Session = Depends(get_session)):
    result = session.exec(
        select(Video, Channel.title, Channel.handle)
        .join(Channel, Video.channel_id == Channel.channel_id)
        .where(Video.video_id == video_id)
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Video not found")
    video, ch_title, ch_handle = result
    data = VideoWithChannel(
        **video.model_dump(),
        channel_title=ch_title,
        channel_handle=ch_handle,
    )
    return {"ok": True, "data": data}
