# Transcript retrieval routes for data pipeline.
# Response shape will align with final schema when set.
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client

from app.core.config import settings

router = APIRouter()


class VideoWithTranscript(BaseModel):
    video_id: str
    transcript: str | None = None
    title: str | None = None
    created_at: datetime | None = None


@router.get("/videos")
def list_videos():
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        return {"error": "Supabase env vars not set"}
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    res = supabase.table("videos").select("*").execute()
    return {"ok": True, "data": res.data, "count": len(res.data)}


@router.get("/videos/{video_id}", response_model=VideoWithTranscript)
def get_video(video_id: str):
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500, detail="Supabase env vars not set")
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    res = supabase.table("videos").select("*").eq("video_id", video_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Video not found")
    return res.data[0]
