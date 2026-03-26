"""
YouTube Data Ingestion Pipeline.

1. Pull official YouTube metadata via Google's YouTube Data API
2. Semantic filter by public health relevance (LLM-based)
3. Filter by metadata impact
4. Pull transcripts via youtube-transcript-api
5. Pull comments via YouTube Data API
6. Persist to Supabase (videos, transcripts, comments)
"""

import json
import math
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from googleapiclient.discovery import build
from supabase import create_client
from youtube_transcript_api import YouTubeTranscriptApi

from pipelines.llm_insight_generation import OllamaProvider
from pipelines.shared import LLMProvider


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


_SEMANTIC_FILTER_SYSTEM = """You classify YouTube video titles for relevance to PUBLIC HEALTH as a domain.

PUBLIC HEALTH means: population-level health, disease outbreaks, vaccination, epidemiology, nutrition as a population issue, maternal health, environmental health, mental health at community/population level, health policy, public health education, misinformation around treatments/prevention/disease/population health.

EXCLUDE (filter out):
- Athlete injury gossip, sports health updates
- Celebrity illness news unless clearly tied to broader public health discussion
- Vague religious/spiritual healing unless directly framed as public-health-relevant
- Random wellness clickbait not actually relevant to public health topics

Return ONLY valid JSON. For each video, output: {"video_id": "...", "is_relevant": true|false, "reason": "short string", "confidence": 0.0-1.0}"""


def _get_llm_provider_for_filtering() -> LLMProvider:
    """Create LLM provider for semantic filtering from env vars."""
    model = os.environ.get("LLM_MODEL", "llama3")
    return OllamaProvider(model=model)


def _parse_semantic_filter_response(
    raw: str, video_ids: List[str]
) -> Dict[str, dict]:
    """
    Parse LLM classification response. Returns dict mapping video_id -> {is_relevant, reason, confidence}.
    Fails closed: malformed or missing entries are treated as not relevant.
    """
    result: Dict[str, dict] = {}
    text = raw.strip()
    if "```" in text:
        for marker in ("```json", "```"):
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find("```", start)
                text = text[start : end if end >= 0 else None].strip()
                break
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return result

    items = parsed if isinstance(parsed, list) else parsed.get("results", parsed.get("items", []))
    if not isinstance(items, list):
        return result

    for item in items:
        if not isinstance(item, dict):
            continue
        vid = item.get("video_id")
        if not vid or vid not in video_ids:
            continue
        is_rel = item.get("is_relevant", False)
        if not isinstance(is_rel, bool):
            is_rel = str(is_rel).lower() in ("true", "1", "yes")
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        result[vid] = {
            "is_relevant": is_rel,
            "reason": str(item.get("reason", ""))[:200],
            "confidence": conf,
        }
    return result


def filter_videos_by_public_health_relevance(
    video_metadata: List[dict],
    provider: Optional[LLMProvider] = None,
    *,
    batch_size: int = 15,
    min_confidence: float = 0.5,
    verbose: bool = True,
) -> List[dict]:
    """
    Filter candidate videos by semantic relevance to public health.

    Uses an LLM to classify each video's title (and optionally description).
    Returns only metadata for videos classified as relevant with sufficient confidence.
    Fails closed: malformed responses or parse errors exclude the affected videos.
    """
    if not video_metadata:
        return []

    prov = provider or _get_llm_provider_for_filtering()
    kept: List[dict] = []
    video_by_id = {v["id"]: v for v in video_metadata}

    for i in range(0, len(video_metadata), batch_size):
        batch = video_metadata[i : i + batch_size]
        batch_ids = [v["id"] for v in batch]
        titles = [
            v.get("snippet", {}).get("title", "")[:200] or "(no title)"
            for v in batch
        ]
        descriptions = [
            (v.get("snippet", {}).get("description", "") or "")[:300]
            for v in batch
        ]

        user_prompt = "Classify each video for public health relevance. Return a JSON array.\n\n"
        for j, (vid, title, desc) in enumerate(zip(batch_ids, titles, descriptions)):
            user_prompt += f"{j+1}. video_id: {vid}\n   title: {title}\n"
            if desc:
                user_prompt += f"   description: {desc[:150]}...\n" if len(desc) > 150 else f"   description: {desc}\n"
            user_prompt += "\n"

        user_prompt += '\nReturn JSON array: [{"video_id":"...","is_relevant":bool,"reason":"...","confidence":0.0-1.0}, ...]'

        try:
            raw = prov.generate_response(
                system=_SEMANTIC_FILTER_SYSTEM, user_prompt=user_prompt
            )
            classifications = _parse_semantic_filter_response(raw, batch_ids)
            for vid in batch_ids:
                c = classifications.get(vid)
                if c and c.get("is_relevant") and c.get("confidence", 0) >= min_confidence:
                    kept.append(video_by_id[vid])
                elif verbose:
                    reason = c.get("reason", "no classification") if c else "parse skipped"
                    print(f"  [filtered] {vid}: {reason[:80]}")
        except Exception as e:
            if verbose:
                print(f"  [semantic filter batch error] {e}")
            for vid in batch_ids:
                if verbose:
                    print(f"  [filtered] {vid}: batch failed (fail closed)")

    return kept


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


_ISO_DURATION_RE = re.compile(
    r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
)


def _parse_iso8601_duration(raw: str) -> Optional[int]:
    """Parse an ISO 8601 duration like PT1H2M10S into total seconds."""
    m = _ISO_DURATION_RE.match(raw or "")
    if not m:
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _build_video_row(item: dict) -> dict:
    """Map a YouTube videos.list item to a dict matching the Video SQLModel."""
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})

    published_at = datetime.fromisoformat(
        snippet["publishedAt"].replace("Z", "+00:00")
    )

    return {
        "video_id": item["id"],
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle"),
        "title": snippet.get("title", ""),
        "category_id": snippet.get("categoryId"),
        "published_at": published_at.isoformat(),
        "view_count": int(stats.get("viewCount", 0) or 0),
        "like_count": int(stats.get("likeCount", 0) or 0),
        "comment_count": int(stats.get("commentCount", 0) or 0),
        "duration_seconds": _parse_iso8601_duration(content.get("duration")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _upsert_videos(
    sb, video_metadata: List[dict], filtered_ids: List[str], *, verbose: bool = True
) -> int:
    """Build Video rows for filtered IDs and upsert into Supabase."""
    by_id = {v["id"]: v for v in video_metadata}
    rows = [_build_video_row(by_id[vid]) for vid in filtered_ids if vid in by_id]
    if not rows:
        return 0
    sb.table("videos").upsert(rows, on_conflict="video_id").execute()
    if verbose:
        print(f"Upserted {len(rows)} video(s) to Supabase")
    return len(rows)


_ytt_api = YouTubeTranscriptApi()


def _save_transcripts(
    sb, video_ids: List[str], *, verbose: bool = True
) -> int:
    """Fetch transcripts, clean, and persist to Supabase. Returns count saved."""
    saved = 0
    for video_id in video_ids:
        try:
            transcript = _ytt_api.fetch(video_id)
            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in transcript.snippets
            ]
            cleaned = _clean_transcript(segments)
            sb.table("transcripts").delete().eq("video_id", video_id).execute()
            sb.table("transcripts").insert({
                "transcript_id": str(uuid.uuid4()),
                "video_id": video_id,
                "cleaned_transcript_txt": cleaned,
                "raw_transcript_json": {"segments": segments},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
            saved += 1
        except Exception as e:
            if verbose:
                print(f"No transcript for {video_id}: {e}")
    return saved


def _save_comments(
    sb, youtube, video_ids: List[str], *, verbose: bool = True
) -> int:
    """Fetch comments and persist to Supabase. Returns count saved."""
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
            sb.table("comments").upsert(
                {
                    "video_id": video_id,
                    "comment_threads_json": {"items": comment_items},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="video_id",
            ).execute()
            saved += 1
        except Exception as e:
            if verbose:
                print(f"No comments for {video_id}: {e}")
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
    filter by semantic relevance and impact, then persist videos,
    transcripts, and comments to Supabase.

    Returns:
        dict with keys: video_ids, videos_upserted, transcripts_saved, comments_saved
    """
    load_dotenv()
    api_key = os.getenv("YOUTUBE_DATA_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_DATA_API_KEY not set in environment")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    sb = create_client(supabase_url, supabase_key)
    youtube = build(serviceName="youtube", version="v3", developerKey=api_key)

    video_ids = _fetch_candidate_video_ids(
        youtube, search_query=search_query, max_search_pages=max_search_pages
    )
    if verbose:
        print(f"Search results: {len(video_ids)} candidates")

    video_metadata = _fetch_video_metadata(youtube, video_ids)
    video_metadata = filter_videos_by_public_health_relevance(
        video_metadata, verbose=verbose
    )
    if verbose:
        print(f"After semantic filter: {len(video_metadata)} public-health-relevant")

    filtered_ids = _filter_by_impact(
        video_metadata,
        min_comments_per_1k=min_comments_per_1k,
        min_likes_per_1k=min_likes_per_1k,
        min_views=min_views,
        percentile=percentile,
    )
    if verbose:
        print(f"After impact filter: {len(filtered_ids)} high-impact")

    videos_upserted = _upsert_videos(sb, video_metadata, filtered_ids, verbose=verbose)
    transcripts_saved = _save_transcripts(sb, filtered_ids, verbose=verbose)
    comments_saved = _save_comments(sb, youtube, filtered_ids, verbose=verbose)

    if verbose:
        print(
            f"Pipeline complete: {videos_upserted} videos, "
            f"{transcripts_saved} transcripts, {comments_saved} comments"
        )

    return {
        "video_ids": filtered_ids,
        "videos_upserted": videos_upserted,
        "transcripts_saved": transcripts_saved,
        "comments_saved": comments_saved,
    }


if __name__ == "__main__":
    run_youtube_data_ingestion_pipeline()
