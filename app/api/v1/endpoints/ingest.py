# Data ingestion pipeline: YouTube API + transcript → channels, videos, transcripts.
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from app.core.config import settings

router = APIRouter()

YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"


def _fetch_video_metadata(video_id: str) -> dict[str, Any]:
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": settings.YOUTUBE_API_KEY,
    }
    with httpx.Client() as client:
        resp = client.get(YOUTUBE_VIDEOS_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("items"):
        raise ValueError("Video not found")
    return data["items"][0]


def _fetch_channel_metadata(channel_id: str) -> dict[str, Any]:
    params = {
        "part": "snippet",
        "id": channel_id,
        "key": settings.YOUTUBE_API_KEY,
    }
    with httpx.Client() as client:
        resp = client.get(YOUTUBE_CHANNELS_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("items"):
        return {"title": "", "handle": "", "url": f"https://www.youtube.com/channel/{channel_id}"}
    sn = data["items"][0].get("snippet", {})
    return {
        "title": sn.get("title", ""),
        "handle": sn.get("customUrl", "") or "",
        "url": f"https://www.youtube.com/channel/{channel_id}",
    }


def _fetch_transcript(video_id: str) -> tuple[str, str]:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcripts disabled for this video")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video unavailable")
    text = " ".join(item["text"] for item in transcript_list)
    return text, "en"


class IngestVideoRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")


@router.post("/ingest/video")
def ingest_video(payload: IngestVideoRequest):
    video_id = payload.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id required")

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    if not settings.YOUTUBE_API_KEY:
        raise HTTPException(status_code=503, detail="YOUTUBE_API_KEY not set")

    try:
        video_resource = _fetch_video_metadata(video_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="YouTube API error")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    snippet = video_resource.get("snippet", {})
    statistics = video_resource.get("statistics", {})
    channel_id = snippet.get("channelId") or ""
    if not channel_id:
        raise HTTPException(status_code=422, detail="Video has no channelId")

    channel_meta = _fetch_channel_metadata(channel_id)
    transcript_text, language = _fetch_transcript(video_id)

    sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    # 1. Upsert channel
    sb.table("channels").upsert(
        {
            "channel_id": channel_id,
            "title": channel_meta["title"],
            "handle": channel_meta["handle"],
            "url": channel_meta["url"],
        },
        on_conflict="channel_id",
    ).execute()

    # 2. Upsert video
    published_at = snippet.get("publishedAt")
    thumbnails = snippet.get("thumbnails", {}) or {}
    thumb_url = (thumbnails.get("default") or thumbnails.get("medium") or {}).get("url") or ""
    view_count = int(statistics.get("viewCount", 0) or 0)
    stats_value = statistics if isinstance(statistics, dict) else None

    sb.table("videos").upsert(
        {
            "video_id": video_id,
            "channel_id": channel_id,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "view_count": view_count,
            "published_at": published_at,
            "thumbnail_url": thumb_url or None,
            "stats": stats_value,
        },
        on_conflict="video_id",
    ).execute()

    # 3. Insert transcript (one row per video; replace if re-ingesting)
    # Delete existing transcripts for this video then insert (idempotent re-run)
    sb.table("transcripts").delete().eq("video_id", video_id).execute()
    sb.table("transcripts").insert(
        {"video_id": video_id, "language": language, "content": transcript_text}
    ).execute()

    return {
        "ok": True,
        "video_id": video_id,
        "channel_id": channel_id,
        "message": "Ingested channel, video, and transcript",
    }
