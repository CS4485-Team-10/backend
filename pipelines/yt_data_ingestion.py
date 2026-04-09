"""
YouTube Data Ingestion Pipeline.

1. Pull official YouTube metadata via Google's YouTube Data API
2. Semantic filter by public health relevance (LLM-based)
3. Filter by metadata impact
4. Pull transcripts via youtube-transcript-api
5. Persist to Supabase (videos, transcripts)
"""

import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from supabase import create_client
from youtube_transcript_api import YouTubeTranscriptApi

# Running `python pipelines/this_file.py` puts `pipelines/` first on sys.path, so
# `import pipelines.*` fails. Prepend backend root so package imports resolve.
_backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from pipelines.llm_insight_generation import OllamaProvider
from pipelines.shared import LLMProvider

# YouTube Data API v3 quota unit costs per request.
# Source: https://developers.google.com/youtube/v3/determine_quota_cost
_QUOTA_COST_SEARCH_LIST = 100
_QUOTA_COST_VIDEOS_LIST = 1

_DEFAULT_QUOTA_BUDGET = 9000  # keep a 1000-unit buffer under the 10,000/day default

# Tuned for ~300+ deduplicated candidate IDs with 6-month window and default pagination.
_DEFAULT_SEARCH_QUERIES = [
    "personal mental health",
    "personal fitness health journey",
    "personal nutrition diet health",
    "sleep health tips",
    "chronic illness management",
    "pregnancy health personal",
    "preventive health wellness",
]


class QuotaBudget:
    """Tracks YouTube Data API quota unit consumption against a budget.

    Call ``try_consume`` before every API request.  When the budget is
    exhausted the method returns ``False`` and records the breach context
    so callers can stop gracefully.
    """

    def __init__(self, budget: int = _DEFAULT_QUOTA_BUDGET):
        self.budget = budget
        self.used = 0
        self.breached = False
        self.breach_stage: Optional[str] = None
        self.breach_detail: Optional[str] = None

    @property
    def remaining(self) -> int:
        return max(self.budget - self.used, 0)

    def try_consume(self, cost: int, *, stage: str = "", detail: str = "") -> bool:
        """Attempt to consume *cost* units.  Returns True on success."""
        if self.used + cost > self.budget:
            if not self.breached:
                self.breached = True
                self.breach_stage = stage
                self.breach_detail = detail
            return False
        self.used += cost
        return True

    def summary(self) -> str:
        status = "QUOTA_BREACH" if self.breached else "OK"
        msg = f"[quota] {status}: used={self.used}/{self.budget} units"
        if self.breached:
            msg += f"; breach_stage={self.breach_stage}; detail={self.breach_detail}"
        return msg


def _chunk_video_ids(lst: List[str], n: int):
    """Break video_ids into smaller chunks for API request limits."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _reraise_if_youtube_quota_exceeded(err: HttpError) -> None:
    """Turn opaque 403 quota errors into a short, actionable message."""
    if err.resp.status != 403:
        raise err
    body = (err.content or b"").decode("utf-8", errors="replace")
    if "quotaExceeded" in body or "youtube.quota" in body:
        raise RuntimeError(
            "YouTube Data API quota exceeded for this API key/project "
            "(Google daily cap, separate from this script's QuotaBudget). "
            "Wait for reset (midnight Pacific), raise quota in Cloud Console, "
            "or reduce search_queries / max_search_pages."
        ) from err
    raise err


def _fetch_video_metadata(
    youtube, video_ids: List[str], budget: QuotaBudget
) -> List[dict]:
    """Fetch hydrated video metrics (snippet, statistics, contentDetails) for given IDs."""
    all_items: List[dict] = []
    for job in _chunk_video_ids(video_ids, 50):
        if not budget.try_consume(
            _QUOTA_COST_VIDEOS_LIST,
            stage="videos.list",
            detail=f"chunk starting {job[0]}",
        ):
            break
        try:
            resp = (
                youtube.videos()
                .list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(job),
                    maxResults=50,
                )
                .execute()
            )
        except HttpError as e:
            _reraise_if_youtube_quota_exceeded(e)
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
        "like_count": like_count,
        "comment_count": comment_count,
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


_SEMANTIC_FILTER_SYSTEM = """You classify YouTube videos for relevance to PUBLIC HEALTH.

PUBLIC HEALTH includes:
- Population-level health topics (disease, prevention, epidemiology)
- Medical or biological explanations of health conditions
- Evidence-based health advice or risk factors
- Mental health topics IF they are educational or generalizable
- Healthcare systems, policy, access, or funding
- Health misinformation or claims about treatments

STRICTLY EXCLUDE:
- Personal life stories or anecdotes (even if health-related)
- Emotional experiences without generalizable insight
- Motivational or self-help content
- Individual-specific situations (e.g., “my experience with…”, “the speaker…”)
- Vague wellness claims without clear mechanism or evidence
- Non-health content (e.g., hobbies, possessions, general life updates)

IMPORTANT:
A video is ONLY relevant if the content is GENERALIZABLE and provides information useful beyond a single individual.

Return ONLY valid JSON in this format:
[
  {
    "video_id": "...",
    "is_relevant": true | false,
    "reason": "short explanation",
    "confidence": 0.0-1.0
  }
]
"""

def _get_llm_provider_for_filtering() -> LLMProvider:
    """Create LLM provider for semantic filtering from YT_SEMANTIC_FILTER_MODEL (default gemma2)."""
    model = os.environ.get("YT_SEMANTIC_FILTER_MODEL", "gemma2")
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
    budget: QuotaBudget,
    *,
    search_queries: List[str] = _DEFAULT_SEARCH_QUERIES,
    max_search_pages: int = 10,
    verbose: bool = True,
) -> List[str]:
    """Fetch candidate video IDs by fanning out across multiple search queries.

    Each query is independently paginated.  Results are deduplicated so
    overlapping queries don't inflate the candidate set.
    """
    six_months_ago = (
        datetime.now(timezone.utc) - timedelta(days=180)
    ).strftime("%Y-%m-%dT00:00:00Z")

    seen_ids: set[str] = set()

    for query in search_queries:
        if budget.breached:
            break

        query_count = 0
        page_token = None
        for page_num in range(max_search_pages):
            if not budget.try_consume(
                _QUOTA_COST_SEARCH_LIST,
                stage="search.list",
                detail=f"q={query!r} page {page_num + 1}/{max_search_pages}",
            ):
                break
            try:
                resp = (
                    youtube.search()
                    .list(
                        q=query,
                        part="snippet",
                        type="video",
                        maxResults=50,
                        publishedAfter=six_months_ago,
                        order="viewCount",
                        relevanceLanguage="en",
                        pageToken=page_token,
                    )
                    .execute()
                )
            except HttpError as e:
                _reraise_if_youtube_quota_exceeded(e)
            for item in resp.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid and item.get("id", {}).get("kind") == "youtube#video":
                    if vid not in seen_ids:
                        seen_ids.add(vid)
                        query_count += 1
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        if verbose:
            print(f"  query {query!r}: {query_count} new IDs (total {len(seen_ids)})")

    return list(seen_ids)


def _filter_by_impact(
    video_metadata: List[dict],
    *,
    min_comments_per_1k: float = 1.0,
    min_likes_per_1k: float = 10,
    min_views: int = 500,
    min_like_count: int = 25,
    min_comment_count: int = 5,
    percentile: float = 0.75,
    verbose: bool = False,
) -> List[str]:
    """Compute impact features, filter by engagement thresholds, return top percentile video IDs.

    Raw minimums (views, likes, comments) block small-sample false positives: per-1k
    ratios alone can look strong on very low view/engagement counts.
    """
    impact_metrics = [_compute_impact_features(v) for v in video_metadata]
    n = len(impact_metrics)
    raw_excluded = sum(
        1
        for v in impact_metrics
        if not (
            v["view_count"] >= min_views
            and v["like_count"] >= min_like_count
            and v["comment_count"] >= min_comment_count
        )
    )
    if verbose and n:
        print(
            f"  [impact] raw floor excluded {raw_excluded}/{n} candidates "
            f"(before ratio+percentile)"
        )

    eligible = [
        v
        for v in impact_metrics
        if (
            v["view_count"] >= min_views
            and v["like_count"] >= min_like_count
            and v["comment_count"] >= min_comment_count
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
    """Fetch transcripts, clean, and persist to Supabase. Returns count saved.

    Skips videos that already have a transcript row (avoids FK conflicts
    when claims reference the existing transcript_id).
    """
    existing = (
        sb.table("transcripts")
        .select("video_id")
        .in_("video_id", video_ids)
        .execute()
    )
    already_have = {r["video_id"] for r in existing.data}

    saved = 0
    for video_id in video_ids:
        if video_id in already_have:
            if verbose:
                print(f"Transcript already exists for {video_id}, skipping")
            continue
        try:
            transcript = _ytt_api.fetch(video_id)
            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in transcript.snippets
            ]
            cleaned = _clean_transcript(segments)

            if not cleaned and not segments:
                if verbose:
                    print(f"Empty transcript for {video_id}, skipping")
                continue

            sb.table("transcripts").insert({
                "transcript_id": str(uuid.uuid4()),
                "video_id": video_id,
                "cleaned_transcript_txt": cleaned or "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
            saved += 1
        except Exception as e:
            if verbose:
                print(f"No transcript for {video_id}: {e}")
    return saved


def run_youtube_data_ingestion_pipeline(
    *,
    search_queries: Optional[List[str]] = None,
    max_search_pages: int = 10,
    min_comments_per_1k: float = 1.0,
    min_likes_per_1k: float = 10,
    min_views: int = 500,
    min_like_count: int = 25,
    min_comment_count: int = 5,
    percentile: float = 0.75,
    quota_budget: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Main entrypoint for the YouTube data ingestion pipeline.

    Coordinates the full workflow: search for videos, fetch metadata,
    filter by semantic relevance and impact, then persist videos and
    transcripts to Supabase.

    Args:
        search_queries: List of search terms to fan out across.
            Defaults to _DEFAULT_SEARCH_QUERIES (~10 public-health terms).
        min_like_count / min_comment_count: Raw engagement floors for the impact
            gate (with min_views), avoiding high per-1k ratios on tiny samples.
        quota_budget: Max YouTube API quota units this run may consume.
            Defaults to env var YT_QUOTA_DAILY_BUDGET_UNITS or 9000.

    Returns:
        dict with keys: video_ids, videos_upserted, transcripts_saved,
        quota_used, quota_budget, quota_breached
    """
    load_dotenv()
    api_key = os.getenv("YOUTUBE_DATA_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_DATA_API_KEY not set in environment")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    budget_limit = quota_budget or int(
        os.getenv("YT_QUOTA_DAILY_BUDGET_UNITS", str(_DEFAULT_QUOTA_BUDGET))
    )
    budget = QuotaBudget(budget=budget_limit)

    sb = create_client(supabase_url, supabase_key)
    youtube = build(serviceName="youtube", version="v3", developerKey=api_key)

    queries = search_queries or _DEFAULT_SEARCH_QUERIES
    video_ids = _fetch_candidate_video_ids(
        youtube,
        budget,
        search_queries=queries,
        max_search_pages=max_search_pages,
        verbose=verbose,
    )
    if verbose:
        print(f"Search results: {len(video_ids)} candidates")

    video_metadata: List[dict] = []
    filtered_ids: List[str] = []
    videos_upserted = 0
    transcripts_saved = 0

    if not budget.breached:
        video_metadata = _fetch_video_metadata(youtube, video_ids, budget)
        video_metadata = filter_videos_by_public_health_relevance(
            video_metadata, verbose=verbose
        )
        if verbose:
            print(f"After semantic filter: {len(video_metadata)} public-health-relevant")

    if not budget.breached:
        filtered_ids = _filter_by_impact(
            video_metadata,
            min_comments_per_1k=min_comments_per_1k,
            min_likes_per_1k=min_likes_per_1k,
            min_views=min_views,
            min_like_count=min_like_count,
            min_comment_count=min_comment_count,
            percentile=percentile,
            verbose=verbose,
        )
        if verbose:
            print(f"After impact filter: {len(filtered_ids)} high-impact")

    if not budget.breached and filtered_ids:
        videos_upserted = _upsert_videos(
            sb, video_metadata, filtered_ids, verbose=verbose
        )
        transcripts_saved = _save_transcripts(sb, filtered_ids, verbose=verbose)

    if verbose:
        print(
            f"Pipeline complete: {videos_upserted} videos, "
            f"{transcripts_saved} transcripts"
        )
        print(budget.summary())

    return {
        "video_ids": filtered_ids,
        "videos_upserted": videos_upserted,
        "transcripts_saved": transcripts_saved,
        "quota_used": budget.used,
        "quota_budget": budget.budget,
        "quota_breached": budget.breached,
    }

if __name__ == "__main__":
    run_youtube_data_ingestion_pipeline()
