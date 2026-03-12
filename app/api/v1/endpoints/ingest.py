import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client

from app.core.config import settings
from app.pipelines.yt_ingest import TranscriptErrors, run_pipeline

router = APIRouter()


class IngestVideoRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")


@router.post("/ingest/video")
def ingest_video(payload: IngestVideoRequest):
    video_id = payload.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id required")

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    api_key = settings.youtube_api_key
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="YOUTUBE_API_KEY or YOUTUBE_DATA_API_KEY not set",
        )

    try:
        payload = run_pipeline(api_key, video_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code, detail="YouTube API error"
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except TranscriptErrors as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    channel_id = payload["channel"]["channel_id"]

    # 1. Upsert channel (Supabase store — notebook had placeholder for this)
    sb.table("channels").upsert(
        payload["channel"],
        on_conflict="channel_id",
    ).execute()

    # 2. Upsert video
    sb.table("videos").upsert(
        payload["video"],
        on_conflict="video_id",
    ).execute()

    # 3. Transcript: delete then insert
    # (idempotent re-run; notebook had placeholder for DB store)
    sb.table("transcripts").delete().eq("video_id", video_id).execute()
    sb.table("transcripts").insert(payload["transcript"]).execute()

    return {
        "ok": True,
        "video_id": video_id,
        "channel_id": channel_id,
        "message": "Ingested channel, video, and transcript",
    }
