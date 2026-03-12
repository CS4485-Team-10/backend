"""
YouTube Data Ingestion Pipeline.

1. Pull official YouTube metadata via Google's YouTube Data API
2. Pull transcripts via youtube-transcript-api
3. Pull comments via YouTube Data API
"""

import json
import math
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Union

from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

# Resolve to backend/data (works when run from backend/ or pipelines/)
DATA_ROOT = (
    Path.cwd().parent / "data"
    if Path.cwd().name == "pipelines"
    else Path.cwd() / "data"
)


def _clear_data_dirs() -> None:
    """Clear subdirectories: metadata, transcripts/cleaned, transcripts/raw, comments."""
    dirs = [
        DATA_ROOT / "metadata",
        DATA_ROOT / "transcripts" / "cleaned",
        DATA_ROOT / "transcripts" / "raw",
        DATA_ROOT / "comments",
    ]
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _chunk_video_ids(lst: List[str], n: int):
    """Break video_ids into smaller chunks for API request limits."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _fetch_video_metadata(youtube, video_ids: List[str]) -> List[dict]:
    """Fetch hydrated video metrics (snippet, statistics, contentDetails) for given IDs."""
    all_items: List[dict] = []
    for job in _chunk_video_ids(video_ids, 50):
        resp = (
            youtube.videos()
            .list(
                part="snippet,statistics,contentDetails",
                id=",".join(job),
                maxResults=50,
            )
            .execute()
        )
        all_items.extend(resp.get("items", []))
    return all_items


def _compute_impact_features(vid_metadata: dict) -> dict:
    """Compute derived impact features from a video metadata dict."""
    stats = vid_metadata.get("statistics", {})
    snippet = vid_metadata.get("snippet", {})

    view_count = int(stats.get("viewCount", 0))
    comment_count = int(stats.get("commentCount", 0))
    like_count = int(stats.get("likeCount", 0))
    published_at = datetime.fromisoformat(
        snippet["publishedAt"].replace("Z", "+00:00")
    )
    days_since = max(
        (datetime.now(timezone.utc) - published_at).days,
        1,
    )

    views_per_day = view_count / days_since
    comments_per_1kviews = (
        (comment_count / view_count) * 1000 if view_count > 0 else 0
    )
    likes_per_1kviews = (like_count / view_count) * 1000 if view_count > 0 else 0

    reach = math.log10(view_count + 1)
    momentum = math.log10(views_per_day + 1)
    engagement = math.log10(comments_per_1kviews + 1) + math.log10(
        likes_per_1kviews + 1
    )
    impact_score = 0.45 * reach + 0.35 * momentum + 0.20 * engagement

    return {
        "video_id": vid_metadata["id"],
        "view_count": view_count,
        "views_per_day": views_per_day,
        "comments_per_1kviews": comments_per_1kviews,
        "likes_per_1kviews": likes_per_1kviews,
        "impact_score": impact_score,
    }


def _filter_ws_by_percentile(
    scored_videos: List[dict], percentile: float = 0.9
) -> List[dict]:
    """Keep videos above the given percentile of impact_score (e.g. 0.9 = top 10%)."""
    if not scored_videos:
        return []
    scored_videos = sorted(
        scored_videos, key=lambda x: x["impact_score"], reverse=True
    )
    cutoff_index = min(
        int(len(scored_videos) * percentile),
        len(scored_videos) - 1,
    )
    threshold_score = scored_videos[cutoff_index]["impact_score"]
    return [v for v in scored_videos if v["impact_score"] >= threshold_score]


def _clean_transcript(transcript_data: Union[List[Dict], str]) -> str:
    """Clean a YouTube transcript by removing noise and formatting artifacts."""
    if isinstance(transcript_data, str):
        text = transcript_data
    else:
        text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in transcript_data
        )
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"(?:^|\s)(?:[A-Z][a-z]*(?:\s+[A-Z][a-z]*)?)\s*:\s*", " ", text
    )
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


def _fetch_candidate_video_ids(
    youtube,
    *,
    search_query: str = "personal health",
    max_search_pages: int = 10,
) -> List[str]:
    """Fetch candidate video IDs from YouTube search API (paginated)."""
    six_months_ago = (
        datetime.now(timezone.utc) - timedelta(days=180)
    ).strftime("%Y-%m-%dT00:00:00Z")

    all_items: List[dict] = []
    page_token = None
    for _ in range(max_search_pages):
        resp = (
            youtube.search()
            .list(
                q=search_query,
                part="snippet",
                type="video",
                maxResults=50,
                publishedAfter=six_months_ago,
                order="viewCount",
                pageToken=page_token,
            )
            .execute()
        )
        all_items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return [
        item["id"]["videoId"]
        for item in all_items
        if item.get("id", {}).get("kind") == "youtube#video"
    ]


def _filter_by_impact(
    video_metadata: List[dict],
    *,
    min_comments_per_1k: float = 2.5,
    min_likes_per_1k: float = 25,
    min_views: int = 1000,
    percentile: float = 0.9,
) -> List[str]:
    """Compute impact features, filter by engagement thresholds, return top percentile video IDs."""
    impact_metrics = [_compute_impact_features(v) for v in video_metadata]
    eligible = [
        v
        for v in impact_metrics
        if (
            v["view_count"] >= min_views
            and v["comments_per_1kviews"] >= min_comments_per_1k
            and v["likes_per_1kviews"] >= min_likes_per_1k
        )
    ]
    high_impact = _filter_ws_by_percentile(eligible, percentile=percentile)
    return [v["video_id"] for v in high_impact]


def _save_transcripts(
    video_ids: List[str],
    formatter: JSONFormatter,
    *,
    verbose: bool = True,
) -> int:
    """Fetch raw transcripts, clean, and save to raw/ and cleaned/ directories. Returns count saved."""
    (DATA_ROOT / "transcripts" / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "transcripts" / "cleaned").mkdir(parents=True, exist_ok=True)
    saved = 0
    for video_id in video_ids:
        try:
            raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
            json_formatted = formatter.format_transcript(raw_transcript, indent=2)
            (DATA_ROOT / "transcripts" / "raw" / f"{video_id}.json").write_text(
                json_formatted, encoding="utf-8"
            )
            cleaned = _clean_transcript(raw_transcript)
            (DATA_ROOT / "transcripts" / "cleaned" / f"{video_id}.txt").write_text(
                cleaned, encoding="utf-8"
            )
            saved += 1
        except Exception as e:
            if verbose:
                print(f"No transcript for {video_id}: {e}")
    return saved


def _save_comments(youtube, video_ids: List[str]) -> int:
    """Fetch comments and save to comments/ directory. Returns count saved."""
    (DATA_ROOT / "comments").mkdir(parents=True, exist_ok=True)
    saved = 0
    for video_id in video_ids:
        try:
            comment_items: List[dict] = []
            page_token = None
            while True:
                resp = (
                    youtube.commentThreads()
                    .list(
                        part="snippet,replies",
                        videoId=video_id,
                        order="relevance",
                        maxResults=100,
                        pageToken=page_token,
                    )
                    .execute()
                )
                comment_items.extend(resp.get("items", []))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            (DATA_ROOT / "comments" / f"{video_id}.json").write_text(
                json.dumps({"items": comment_items}, indent=2),
                encoding="utf-8",
            )
            saved += 1
        except Exception:
            pass
    return saved


def run_youtube_data_ingestion_pipeline(
    *,
    search_query: str = "personal health",
    max_search_pages: int = 10,
    min_comments_per_1k: float = 2.5,
    min_likes_per_1k: float = 25,
    min_views: int = 1000,
    percentile: float = 0.9,
    verbose: bool = True,
) -> dict:
    """
    Main entrypoint for the YouTube data ingestion pipeline.

    Coordinates the full workflow: search for videos, fetch metadata,
    filter by impact, save transcripts and comments. Call this from a backend
    route or run locally via `python -m pipelines.yt_data_ingestion`.

    Returns:
        dict with keys: video_ids, transcripts_saved, comments_saved
    """
    load_dotenv()
    api_key = os.getenv("YOUTUBE_DATA_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_DATA_API_KEY not set in environment")

    _clear_data_dirs()
    youtube = build(serviceName="youtube", version="v3", developerKey=api_key)

    video_ids = _fetch_candidate_video_ids(
        youtube, search_query=search_query, max_search_pages=max_search_pages
    )
    video_metadata = _fetch_video_metadata(youtube, video_ids)
    filtered_ids = _filter_by_impact(
        video_metadata,
        min_comments_per_1k=min_comments_per_1k,
        min_likes_per_1k=min_likes_per_1k,
        min_views=min_views,
        percentile=percentile,
    )

    if verbose:
        print(f"Candidates: {len(video_ids)}, Filtered: {len(filtered_ids)}")

    formatter = JSONFormatter()
    transcripts_saved = _save_transcripts(
        filtered_ids, formatter, verbose=verbose
    )
    comments_saved = _save_comments(youtube, filtered_ids)

    if verbose:
        raw_count = sum(
            1
            for p in (DATA_ROOT / "transcripts" / "raw").iterdir()
            if p.is_file()
        )
        cleaned_count = sum(
            1
            for p in (DATA_ROOT / "transcripts" / "cleaned").iterdir()
            if p.is_file()
        )
        if raw_count == cleaned_count:
            print(f"Verification: {raw_count} files in both directories.")
        else:
            print(f"Mismatch: {raw_count} raw, {cleaned_count} cleaned")

    return {
        "video_ids": filtered_ids,
        "transcripts_saved": transcripts_saved,
        "comments_saved": comments_saved,
    }


if __name__ == "__main__":
    run_youtube_data_ingestion_pipeline()
