from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, func, select

from app.core.database import get_session
from app.models.channel import Channel

router = APIRouter()


@router.get("/channels")
def list_channels(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    total = session.exec(select(func.count()).select_from(Channel)).one()
    channels = session.exec(select(Channel).offset(skip).limit(limit)).all()
    return {"ok": True, "data": channels, "count": total}


@router.get("/channels/{channel_id}")
def get_channel(channel_id: str, session: Session = Depends(get_session)):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return {"ok": True, "data": channel}
