# YouTube data ingestion pipeline — extracted from yt_data_ingestion.ipynb (DS team).
# Fetches video metadata (YouTube Data API), channel metadata,
# and transcript; cleans transcript per notebook.
from __future__ import annotations

import re
from typing import Any

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"


def fetch_video_metadata(api_key: str, video_id: str) -> dict[str, Any]:
    """Get hydrated video metadata (snippet, statistics, contentDetails).

    Matches notebook fetch_video_metadata.
    """
    params = {
        "part": "snippet,statistics,contentDetails",
        "id": video_id,
        "key": api_key,
    }
    with httpx.Client() as client:
        resp = client.get(YOUTUBE_VIDEOS_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("items"):
        raise ValueError("Video not found")
    return data["items"][0]


def fetch_channel_metadata(api_key: str, channel_id: str) -> dict[str, Any]:
    """Get channel snippet (title, handle, url). Used for DB channel row."""
    params = {
        "part": "snippet",
        "id": channel_id,
        "key": api_key,
    }
    with httpx.Client() as client:
        resp = client.get(YOUTUBE_CHANNELS_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("items"):
        return {
            "title": "",
            "handle": "",
            "url": f"https://www.youtube.com/channel/{channel_id}",
        }
    sn = data["items"][0].get("snippet", {})
    return {
        "title": sn.get("title", ""),
        "handle": sn.get("customUrl", "") or "",
        "url": f"https://www.youtube.com/channel/{channel_id}",
    }


def clean_transcript(transcript_data: list[dict[str, Any]]) -> str:
    """
    Clean a YouTube transcript by removing noise and formatting artifacts.
    Extracted from yt_data_ingestion.ipynb (clean_transcript).
    """
    text = " ".join(item.get("text", "") for item in transcript_data)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?:^|\s)(?:[A-Z][a-z]*(?:\s+[A-Z][a-z]*)?)\s*:\s*", " ", text)
    text = re.sub(r"\[[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", "", text)
    for pattern in [
        r"\b(?:um|uh|ugh|hmm)\b",
        r"\byou\s+know\b",
        r"\blike\b(?=\s+(?:he|she|they|it|the|a|i))",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?])\s+(?=[a-z])", lambda m: m.group(1) + " ", text)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def fetch_transcript(video_id: str) -> tuple[str, str]:
    """Fetch transcript and return (cleaned_text, language).

    Raises transcript-api errors.
    """
    raw = YouTubeTranscriptApi.get_transcript(video_id)
    cleaned = clean_transcript(raw)
    return cleaned, "en"


def run_pipeline(api_key: str, video_id: str) -> dict[str, Any]:
    """
    Run the DS ingestion pipeline for one video.

    Fetch video + channel + transcript (cleaned).
    Returns payload for backend to write to Supabase (channels, videos, transcripts).
    """
    video_resource = fetch_video_metadata(api_key, video_id)
    snippet = video_resource.get("snippet", {})
    statistics = video_resource.get("statistics", {})
    channel_id = snippet.get("channelId") or ""
    if not channel_id:
        raise ValueError("Video has no channelId")

    channel_meta = fetch_channel_metadata(api_key, channel_id)
    transcript_text, language = fetch_transcript(video_id)

    published_at = snippet.get("publishedAt")
    thumbnails = snippet.get("thumbnails", {}) or {}
    thumb_url = (thumbnails.get("default") or thumbnails.get("medium") or {}).get(
        "url"
    ) or ""
    view_count = int(statistics.get("viewCount", 0) or 0)

    return {
        "channel": {
            "channel_id": channel_id,
            "title": channel_meta["title"],
            "handle": channel_meta["handle"],
            "url": channel_meta["url"],
        },
        "video": {
            "video_id": video_id,
            "channel_id": channel_id,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "view_count": view_count,
            "published_at": published_at,
            "thumbnail_url": thumb_url or None,
            "stats": statistics if isinstance(statistics, dict) else None,
        },
        "transcript": {
            "video_id": video_id,
            "language": language,
            "content": transcript_text,
        },
    }


# Re-export for callers that need to map transcript-api exceptions to HTTP
TranscriptErrors = (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable)
