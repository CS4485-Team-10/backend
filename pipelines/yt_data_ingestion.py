"""Fetch YouTube metadata/transcripts and store videos + transcripts."""

import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from supabase import create_client
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
)

from pipelines.shared import BedrockProvider, LLMProvider, OllamaProvider

# region agent log
_AGENT_LOG_PATH = Path(
    "/Users/advaychandramouli/dev/projects/youtube-intelligence-platform/backend/.cursor/"
    "debug-66acb4.log"
)
_AGENT_DEBUG_SESSION = "66acb4"


def _agent_debug_log(
    hypothesis_id: str, message: str, data: dict, *, run_id: str = "pre-fix"
) -> None:
    try:
        payload: dict = {
            "sessionId": _AGENT_DEBUG_SESSION,
            "runId": run_id,
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "message": message,
            "data": data,
        }
        with _AGENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


# endregion

# YouTube Data API v3 quota unit costs per request.
# Source: https://developers.google.com/youtube/v3/determine_quota_cost
_QUOTA_COST_SEARCH_LIST = 100
_QUOTA_COST_VIDEOS_LIST = 1
_DEFAULT_QUOTA_BUDGET = 9500  # keep a 500-unit buffer under the 10,000/day default
_DEFAULT_SEARCH_QUERIES = [
    "personal mental health",
    "personal fitness health journey",
    "personal nutrition diet health",
    "sleep health tips",
    "chronic illness management",
    "pregnancy health personal",
    "preventive health wellness",
]

# YouTube search.list `order` allowlist.
# Source: https://developers.google.com/youtube/v3/docs/search/list#order
_VALID_SEARCH_ORDERS = {
    "date",
    "rating",
    "relevance",
    "title",
    "videoCount",
    "viewCount",
}

# Default multi-pass strategy: one pass for query relevance, one for popularity.
_DEFAULT_SEARCH_ORDER_PASSES: List[str] = ["relevance", "viewCount"]


class QuotaBudget:
    """Track consumed YouTube API quota against a budget."""

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
        """Consume cost units if still within budget."""
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
    """Yield chunks of IDs."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _reraise_if_youtube_quota_exceeded(err: HttpError) -> None:
    """Raise a clear error for quota-exceeded responses."""
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
    """Fetch snippet/statistics/contentDetails for video IDs."""
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
    """Compute impact features from metadata."""
    stats = vid_metadata.get("statistics", {})
    snippet = vid_metadata.get("snippet", {})

    view_count = int(stats.get("viewCount", 0))
    comment_count = int(stats.get("commentCount", 0))
    like_count = int(stats.get("likeCount", 0))
    published_at = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
    days_since = max(
        (datetime.now(timezone.utc) - published_at).days,
        1,
    )

    views_per_day = view_count / days_since
    comments_per_1kviews = (comment_count / view_count) * 1000 if view_count > 0 else 0
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
    """Keep videos above the impact-score percentile."""
    if not scored_videos:
        return []
    scored_videos = sorted(scored_videos, key=lambda x: x["impact_score"], reverse=True)
    cutoff_index = min(
        int(len(scored_videos) * percentile),
        len(scored_videos) - 1,
    )
    threshold_score = scored_videos[cutoff_index]["impact_score"]
    return [v for v in scored_videos if v["impact_score"] >= threshold_score]


def _clean_transcript(transcript_data: Union[List[Dict], str]) -> str:
    """Clean transcript noise and formatting artifacts."""
    if isinstance(transcript_data, str):
        text = transcript_data
    else:
        text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in transcript_data
        )
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?:^|\s)(?:[A-Z][a-z]*(?:\s+[A-Z][a-z]*)?)\s*:\s*", " ", text)
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


_SEMANTIC_FILTER_SYSTEM = """You classify YouTube videos for relevance to GENERALIZABLE HEALTH INFORMATION.

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

HARD GATE (must satisfy to be relevant):
The video MUST contain at least one of the following:
- explanation of a health concept, mechanism, or condition
- generalizable advice that applies beyond the individual
- discussion of risk factors, causes, or prevention
- structured educational or informational content

If the video is primarily:
- a personal journey (fitness, weight loss, recovery, pregnancy, etc.)
- lifestyle documentation (“day in my life”, routines, habits)
- emotional reflection or storytelling

→ then it is NOT relevant, even if it mentions health topics.

IMPORTANT RETRIEVAL CONTEXT:
These videos were retrieved using health-related search queries. Treat alignment with the search intent as positive evidence of relevance.
Query alignment is weak evidence; the HARD GATE is stronger and must override query alignment.
If a video is about a health-related topic represented by the query context — such as sleep, pregnancy, nutrition, exercise, chronic illness, preventive health, or mental health — it should generally be considered relevant IF the content appears generalizable, educational, explanatory, or broadly informative.
Do NOT require the video to explicitly mention "public health" or be framed at a population-policy level.

However, query alignment alone is NOT sufficient. Still exclude videos that are primarily:
- personal anecdotes or vlogs
- motivational/self-help content
- celebrity/news gossip
- relationship/lifestyle drama
- highly individual situations without broader educational value
- health topics mentioned only incidentally

You must also decide whether the video is ENGLISH / ENGLISH-USABLE for transcript
and downstream claim extraction. Set "is_english_usable" = true if the title and
description suggest the spoken content is primarily English (a few non-English
hashtags, loanwords, or emoji are fine). Set it to false if the content appears
to be primarily in another language.

Return ONLY valid JSON in this format:
[
  {
    "video_id": "...",
    "is_relevant": true | false,
    "is_english_usable": true | false,
    "reason": "short explanation",
    "confidence": 0.0-1.0
  }
]
"""


def _get_llm_provider_for_filtering() -> LLMProvider:
    """Build semantic-filter provider from LLM_PROVIDER/LLM_MODEL environment."""
    provider_name = (os.environ.get("LLM_PROVIDER") or "ollama").lower()

    if provider_name == "ollama":
        model = os.environ.get("OLLAMA_LLM_MODEL") or "gemma2"
        return OllamaProvider(model=model)
    if provider_name == "bedrock":
        raw = (
            os.environ.get("BEDROCK_LLM_MODEL")
            or os.environ.get("LLM_MODEL")
            or os.environ.get("INGEST_LLM_MODEL")
            or ""
        )
        explicit = raw.strip() or None
        return BedrockProvider(model=explicit)
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider_name}. Use 'ollama' or 'bedrock'."
    )


def _parse_json_from_llm_text(text: str) -> Optional[object]:
    """Parse JSON from model output; tolerate leading/trailing prose (e.g. 'Here is…' before `[`)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    return None


def _parse_semantic_filter_response(raw: str, video_ids: List[str]) -> Dict[str, dict]:
    """Parse classification response into {video_id: decision}."""
    result: Dict[str, dict] = {}
    text = raw.strip()
    if "```" in text:
        for marker in ("```json", "```"):
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find("```", start)
                text = text[start : end if end >= 0 else None].strip()
                break
    expected = set(video_ids)
    err_first: Optional[BaseException] = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        err_first = e
        parsed = _parse_json_from_llm_text(text)
        if parsed is not None:
            # region agent log
            _agent_debug_log(
                "H1",
                "json_recovered_after_preamble_strip",
                {
                    "text_len": len(text),
                    "parsed_type": type(parsed).__name__,
                    "first_error": str(e)[:200],
                },
                run_id="post-fix",
            )
            # endregion
    if parsed is None:
        # region agent log
        _agent_debug_log(
            "H1",
            "json_loads_failed",
            {
                "error": str(err_first)[:200] if err_first else "unknown",
                "text_len": len(text),
                "text_head_400": text[:400],
            },
        )
        # endregion
        return result

    items = (
        parsed
        if isinstance(parsed, list)
        else parsed.get("results", parsed.get("items", []))
    )
    if not isinstance(items, list):
        # region agent log
        _agent_debug_log(
            "H2",
            "items_not_list",
            {
                "parsed_type": type(parsed).__name__,
                "parsed_dict_keys": list(parsed.keys())
                if isinstance(parsed, dict)
                else None,
                "items_rejected_type": type(items).__name__,
            },
        )
        # endregion
        return result

    def _coerce_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).lower() in ("true", "1", "yes")

    # region agent log
    non_dict = 0
    mismatch: List[dict] = []
    first_item_keys: Optional[List[str]] = None
    for item in items:
        if not isinstance(item, dict):
            non_dict += 1
            continue
        if first_item_keys is None:
            first_item_keys = list(item.keys())
        raw_id = item.get("video_id", item.get("videoId"))
        cand = raw_id if raw_id is None else str(raw_id).strip()
        if cand and cand not in video_ids and raw_id is not None:
            mismatch.append(
                {
                    "raw_video_id": repr(raw_id)[:120],
                    "coerced": repr(cand)[:120],
                    "in_batch": cand in expected,
                }
            )
    # endregion

    for item in items:
        if not isinstance(item, dict):
            continue
        vid = item.get("video_id")
        if not vid or vid not in video_ids:
            continue
        is_rel = _coerce_bool(item.get("is_relevant"), False)
        # Fail closed for language: missing field => not English-usable.
        is_english_usable = _coerce_bool(item.get("is_english_usable"), False)
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        result[vid] = {
            "is_relevant": is_rel,
            "is_english_usable": is_english_usable,
            "reason": str(item.get("reason", ""))[:200],
            "confidence": conf,
        }

    # region agent log
    missing = list(expected - set(result.keys()))
    llm_ids_seen: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for k in ("video_id", "videoId"):
            if k in item and item[k] is not None:
                llm_ids_seen.append(str(item[k])[:20])
                break
    _agent_debug_log(
        "H3",
        "parse_outcome",
        {
            "n_expected": len(expected),
            "n_items": len(items),
            "n_result": len(result),
            "n_non_dict_items": non_dict,
            "missing_count": len(missing),
            "missing_ids_sample": missing[:30],
            "first_item_keys": first_item_keys,
            "mismatch_guess": mismatch[:20],
            "llm_id_samples": llm_ids_seen[:25],
        },
    )
    # endregion
    return result


def filter_videos_by_public_health_relevance(
    video_metadata: List[dict],
    provider: Optional[LLMProvider] = None,
    *,
    batch_size: int = 15,
    min_confidence: float = 0.5,
    verbose: bool = True,
) -> List[dict]:
    """Keep only public-health-relevant videos from LLM output."""
    if not video_metadata:
        return []

    prov = provider or _get_llm_provider_for_filtering()
    kept: List[dict] = []
    video_by_id = {v["id"]: v for v in video_metadata}

    # region agent log
    all_ids_order = [v["id"] for v in video_metadata]
    all_titles = {
        v["id"]: (v.get("snippet") or {}).get("title", "")[:200] or "(no title)"
        for v in video_metadata
    }
    id_fp = hashlib.md5(
        json.dumps(all_ids_order, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    _agent_debug_log(
        "H4",
        "semantic_filter_cohort",
        {
            "n_videos": len(all_ids_order),
            "ids_fingerprint": id_fp,
            "ids_in_order": all_ids_order,
            "title_by_id": all_titles,
        },
    )
    # endregion

    for i in range(0, len(video_metadata), batch_size):
        batch = video_metadata[i : i + batch_size]
        batch_ids = [v["id"] for v in batch]
        titles = [
            v.get("snippet", {}).get("title", "")[:200] or "(no title)" for v in batch
        ]
        descriptions = [
            (v.get("snippet", {}).get("description", "") or "")[:300] for v in batch
        ]

        user_prompt = (
            "Classify each video for public health relevance. Return a JSON array.\n\n"
        )
        for j, (vid, title, desc) in enumerate(zip(batch_ids, titles, descriptions)):
            user_prompt += f"{j + 1}. video_id: {vid}\n   title: {title}\n"
            if desc:
                user_prompt += (
                    f"   description: {desc[:150]}...\n"
                    if len(desc) > 150
                    else f"   description: {desc}\n"
                )
            user_prompt += "\n"

        user_prompt += '\nReturn JSON array: [{"video_id":"...","is_relevant":bool,"is_english_usable":bool,"reason":"...","confidence":0.0-1.0}, ...]'

        # region agent log
        _agent_debug_log(
            "H5",
            "semantic_filter_batch",
            {
                "batch_index": i // batch_size,
                "batch_size": len(batch_ids),
                "batch_ids": batch_ids,
                "batch_titles": {bid: all_titles.get(bid, "") for bid in batch_ids},
            },
        )
        # endregion

        try:
            raw = prov.generate_response(
                system=_SEMANTIC_FILTER_SYSTEM, user_prompt=user_prompt
            )
            # region agent log
            _agent_debug_log(
                "H5",
                "llm_response_stats",
                {
                    "batch_index": i // batch_size,
                    "raw_len": len(raw or ""),
                    "raw_head_300": (raw or "")[:300],
                },
            )
            # endregion
            classifications = _parse_semantic_filter_response(raw, batch_ids)
            for vid in batch_ids:
                c = classifications.get(vid)
                if (
                    c
                    and c.get("is_relevant")
                    and c.get("is_english_usable")
                    and c.get("confidence", 0) >= min_confidence
                ):
                    kept.append(video_by_id[vid])
                elif verbose:
                    if not c:
                        reason = "parse skipped"
                    elif not c.get("is_english_usable"):
                        reason = "not English-usable"
                    else:
                        reason = c.get("reason", "no classification")
                    print(f"  [filtered] {vid}: {reason}")
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
    search_order_passes: List[str] = _DEFAULT_SEARCH_ORDER_PASSES,
    max_search_pages: int,
    verbose: bool = True,
) -> List[str]:
    """Fetch candidate video IDs across query fan-out with multiple sort-order passes.

    Each element of ``search_order_passes`` drives a full sweep over all
    ``search_queries``.  A single ``seen_ids`` set deduplicates across passes,
    so IDs found in the relevance pass are not re-counted in the viewCount pass.
    The returned list preserves insertion order (relevance pass first).

    Two passes (relevance + viewCount) roughly double search quota; the
    existing QuotaBudget guard stops early on any breach.
    """
    invalid = [o for o in search_order_passes if o not in _VALID_SEARCH_ORDERS]
    if invalid:
        raise ValueError(
            f"Invalid search order(s): {invalid!r}. "
            f"Must be one of {sorted(_VALID_SEARCH_ORDERS)}."
        )

    target_timeframe = (datetime.now(timezone.utc) - timedelta(days=90)).strftime(
        "%Y-%m-%dT00:00:00Z"
    )

    # Insertion-ordered list; seen_ids tracks dedup across all passes.
    candidate_ids: List[str] = []
    seen_ids: set[str] = set()

    for order in search_order_passes:
        if budget.breached:
            break
        if verbose:
            print(f"[search] pass order={order!r}")

        for query in search_queries:
            if budget.breached:
                break

            query_count = 0
            page_token = None
            for page_num in range(max_search_pages):
                if not budget.try_consume(
                    _QUOTA_COST_SEARCH_LIST,
                    stage="search.list",
                    detail=f"order={order!r} q={query!r} page {page_num + 1}/{max_search_pages}",
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
                            publishedAfter=target_timeframe,
                            order=order,
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
                            candidate_ids.append(vid)
                            query_count += 1
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

            if verbose:
                print(
                    f"  [order={order!r}] query {query!r}: "
                    f"{query_count} new IDs (running total {len(candidate_ids)})"
                )

    return candidate_ids


def _filter_by_impact(
    video_metadata: List[dict],
    *,
    min_comments_per_1k: float = 1.0,
    min_likes_per_1k: float = 10,
    min_views: int = 500,
    min_like_count: int = 25,
    min_comment_count: int = 5,
    percentile: float = 0.50,
    verbose: bool = False,
) -> List[str]:
    """Filter by engagement and return top-percentile IDs."""
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


_ISO_DURATION_RE = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _parse_iso8601_duration(raw: str) -> Optional[int]:
    """Parse ISO-8601 duration into seconds."""
    m = _ISO_DURATION_RE.match(raw or "")
    if not m:
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _filter_by_min_duration(
    video_metadata: List[dict],
    *,
    min_duration_seconds: int,
    verbose: bool = True,
) -> List[dict]:
    """Drop videos shorter than ``min_duration_seconds`` (Shorts-style content).

    Unparseable or zero-second durations are treated as failing the gate to
    avoid wasting downstream compute on edge/Shorts metadata.
    """
    if min_duration_seconds <= 0:
        return video_metadata

    kept: List[dict] = []
    excluded = 0
    for item in video_metadata:
        raw = (item.get("contentDetails") or {}).get("duration")
        seconds = _parse_iso8601_duration(raw) or 0
        if seconds >= min_duration_seconds:
            kept.append(item)
        else:
            excluded += 1
    if verbose:
        print(
            f"  [duration] excluded {excluded}/{len(video_metadata)} videos "
            f"shorter than {min_duration_seconds}s"
        )
    return kept


# Unicode script ranges for cheap non-Latin detection on title/description.
_NON_LATIN_RANGES = (
    (0x0400, 0x04FF),  # Cyrillic
    (0x0500, 0x052F),  # Cyrillic Supplement
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0E00, 0x0E7F),  # Thai
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul Syllables
)


def _is_non_latin_letter(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _NON_LATIN_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _heuristic_latin_english_plausible(
    title: str, description: str, *, max_desc_chars: int = 250
) -> bool:
    """Cheap deterministic check that title+description look English-plausible.

    Returns False only when non-Latin letters dominate the metadata, so English
    text with a few hashtags, emoji, or loanwords still passes through.
    """
    text = f"{title or ''} {(description or '')[:max_desc_chars]}"
    latin = 0
    non_latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if _is_non_latin_letter(ch):
            non_latin += 1
        elif ch.isascii() or ch.lower() != ch.upper():
            latin += 1

    total_letters = latin + non_latin
    if total_letters < 8:
        # Too little textual signal; defer to LLM rather than hard-rejecting.
        return True
    if non_latin == 0:
        return True
    if latin < 10 and non_latin >= 5:
        return False
    return (non_latin / total_letters) < 0.4


def _filter_by_language_heuristic(
    video_metadata: List[dict], *, verbose: bool = True
) -> List[dict]:
    """Drop videos whose title/description metadata is non-Latin-script-heavy."""
    kept: List[dict] = []
    excluded = 0
    for item in video_metadata:
        snippet = item.get("snippet", {}) or {}
        title = snippet.get("title", "") or ""
        desc = snippet.get("description", "") or ""
        if _heuristic_latin_english_plausible(title, desc):
            kept.append(item)
            continue
        excluded += 1
        if verbose:
            print(f"  [filtered] {item.get('id')}: non-Latin metadata heuristic")
    if verbose:
        print(
            f"  [language-heuristic] excluded {excluded}/{len(video_metadata)} "
            f"non-Latin-heavy candidates"
        )
    return kept


def _build_video_row(item: dict) -> dict:
    """Map a videos.list item to a DB row payload."""
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})

    published_at = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))

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


def _fetch_all_video_ids(sb, *, page_size: int = 1000) -> set[str]:
    """Return the set of all video_id values already stored in Supabase.

    Paginates with .range() because PostgREST returns at most ``page_size``
    rows per request (default cap is often 1 000).
    """
    known: set[str] = set()
    offset = 0
    while True:
        rows = (
            sb.table("videos")
            .select("video_id")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        for r in rows:
            known.add(r["video_id"])
        if len(rows) < page_size:
            break
        offset += page_size
    return known


def _upsert_videos(
    sb, video_metadata: List[dict], filtered_ids: List[str], *, verbose: bool = True
) -> int:
    """Build video rows and upsert by `video_id`."""
    by_id = {v["id"]: v for v in video_metadata}
    rows = [_build_video_row(by_id[vid]) for vid in filtered_ids if vid in by_id]
    if not rows:
        return 0
    sb.table("videos").upsert(rows, on_conflict="video_id").execute()
    if verbose:
        print(f"Upserted {len(rows)} video(s) to Supabase")
    return len(rows)


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
_ytt_proxy_username = os.getenv("YOUTUBE_TRANSCRIPT_PROXY_USERNAME")
_ytt_proxy_password = os.getenv("YOUTUBE_TRANSCRIPT_PROXY_PASSWORD")
if _ytt_proxy_username and _ytt_proxy_password:
    _ytt_api = YouTubeTranscriptApi(
        proxy_config=WebshareProxyConfig(
            proxy_username=_ytt_proxy_username,
            proxy_password=_ytt_proxy_password,
        )
    )
else:
    _ytt_api = YouTubeTranscriptApi()


def _fetch_transcript_with_fallback(video_id: str):
    """Return ``(fetched, source)`` for a video or ``(None, reason)``.

    ``source`` is one of: ``"en"`` (direct English track), ``"translated:<code>"``
    (translated to English from another language), ``"none"`` (no usable track),
    or ``"blocked"`` (request/IP blocked). The caller logs accordingly.
    """
    try:
        return _ytt_api.fetch(video_id, languages=["en"]), "en"
    except (RequestBlocked, IpBlocked):
        return None, "blocked"
    except NoTranscriptFound:
        pass
    except CouldNotRetrieveTranscript:
        return None, "none"

    try:
        transcript_list = _ytt_api.list(video_id)
    except (RequestBlocked, IpBlocked):
        return None, "blocked"
    except CouldNotRetrieveTranscript:
        return None, "none"

    translatable = [t for t in transcript_list if getattr(t, "is_translatable", False)]
    candidates = translatable or list(transcript_list)
    for transcript in candidates:
        try:
            translated = transcript.translate("en").fetch()
        except (RequestBlocked, IpBlocked):
            return None, "blocked"
        except CouldNotRetrieveTranscript:
            continue
        except Exception:
            continue
        source_lang = getattr(transcript, "language_code", "") or "?"
        return translated, f"translated:{source_lang}"

    return None, "none"


def _save_transcripts(sb, video_ids: List[str], *, verbose: bool = True) -> int:
    """Fetch, clean, and insert transcripts. Return saved count."""
    existing = (
        sb.table("transcripts").select("video_id").in_("video_id", video_ids).execute()
    )
    already_have = {r["video_id"] for r in existing.data}

    saved = 0
    for video_id in video_ids:
        if video_id in already_have:
            if verbose:
                print(f"Transcript already exists for {video_id}, skipping")
            continue
        time.sleep(random.uniform(2, 5))
        try:
            fetched, source = _fetch_transcript_with_fallback(video_id)
            if fetched is None:
                if verbose:
                    if source == "blocked":
                        print(f"Transcript blocked (IP/request) for {video_id}")
                    else:
                        print(f"No usable transcript for {video_id}")
                continue

            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in fetched.snippets
            ]
            cleaned = _clean_transcript(segments)

            if not cleaned and not segments:
                if verbose:
                    print(f"Empty transcript for {video_id}, skipping")
                continue

            sb.table("transcripts").insert(
                {
                    "transcript_id": str(uuid.uuid4()),
                    "video_id": video_id,
                    "cleaned_transcript_txt": cleaned or "",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "processing_status": "pending",
                    "attempt_count": 0,
                }
            ).execute()
            saved += 1
            if verbose:
                if source == "en":
                    print(f"Transcript saved (English direct) for {video_id}")
                elif source.startswith("translated:"):
                    src_lang = source.split(":", 1)[1]
                    print(
                        f"Transcript saved (translated to English from "
                        f"{src_lang}) for {video_id}"
                    )
        except Exception as e:
            if verbose:
                print(f"No transcript for {video_id}: {e}")
    return saved


def run_youtube_data_ingestion_pipeline(
    *,
    search_queries: Optional[List[str]] = None,
    search_order_passes: Optional[List[str]] = None,
    exclude_existing_video_ids: bool = True,
    max_search_pages: int = 3,
    min_comments_per_1k: float = 1.0,
    min_likes_per_1k: float = 10,
    min_views: int = 1500,
    min_like_count: int = 50,
    min_comment_count: int = 10,
    percentile: float = 0.65,
    min_duration_seconds: int = 120,
    quota_budget: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Run ingestion and return IDs, write counts, and quota metrics.

    Order: search (multi-pass) → exclude existing DB IDs → videos.list metadata →
    duration filter → language heuristic → impact filter →
    LLM semantic filter (relevance + English-usable) → upsert + transcripts.

    The semantic filter uses ``_get_llm_provider_for_filtering()``: set
    ``LLM_PROVIDER=bedrock`` (default is ``ollama``) and optionally ``BEDROCK_MODEL``,
    ``LLM_MODEL``, or ``INGEST_LLM_MODEL`` for the Bedrock model ID (see
    ``BedrockProvider`` in ``pipelines.shared.llm_providers``).

    Args:
        search_order_passes: List of YouTube search `order` values to sweep through.
            Defaults to ["relevance", "viewCount"].  Each pass runs the full
            search_queries fan-out; IDs found in an earlier pass are skipped in
            later passes.  Valid values: date, rating, relevance, title,
            videoCount, viewCount.
        exclude_existing_video_ids: When True (default), any video_id already
            present in the `videos` table is removed from the candidate list
            before videos.list and all downstream quota is spent.  Set to False
            for local re-runs where re-processing is acceptable.
    """
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
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
    passes = search_order_passes or _DEFAULT_SEARCH_ORDER_PASSES

    candidate_ids = _fetch_candidate_video_ids(
        youtube,
        budget,
        search_queries=queries,
        search_order_passes=passes,
        max_search_pages=max_search_pages,
        verbose=verbose,
    )
    if verbose:
        print(
            f"Search results: {len(candidate_ids)} unique candidates across all passes"
        )

    # Drop IDs already stored so downstream quota is spent only on new videos.
    existing_skipped = 0
    if exclude_existing_video_ids and candidate_ids:
        existing_ids = _fetch_all_video_ids(sb)
        original_count = len(candidate_ids)
        candidate_ids = [vid for vid in candidate_ids if vid not in existing_ids]
        existing_skipped = original_count - len(candidate_ids)
        if verbose:
            print(
                f"After excluding existing: {len(candidate_ids)} new candidates "
                f"({existing_skipped} already in DB skipped)"
            )
        if not candidate_ids:
            if verbose:
                print(
                    "0 new candidates — all search results already ingested. "
                    "Skipping videos.list and downstream quota."
                )
            print(budget.summary())
            return {
                "video_ids": [],
                "videos_upserted": 0,
                "transcripts_saved": 0,
                "candidates_total": original_count,
                "candidates_new": 0,
                "existing_skipped": existing_skipped,
                "quota_used": budget.used,
                "quota_budget": budget.budget,
                "quota_breached": budget.breached,
            }

    video_metadata: List[dict] = []
    filtered_ids: List[str] = []
    videos_upserted = 0
    transcripts_saved = 0

    if not budget.breached:
        video_metadata = _fetch_video_metadata(youtube, candidate_ids, budget)
        if verbose:
            print(f"Fetched metadata for {len(video_metadata)} videos")

    if not budget.breached and video_metadata:
        video_metadata = _filter_by_min_duration(
            video_metadata,
            min_duration_seconds=min_duration_seconds,
            verbose=verbose,
        )
        if verbose:
            print(f"After duration filter: {len(video_metadata)} videos")

    if not budget.breached and video_metadata:
        video_metadata = _filter_by_language_heuristic(video_metadata, verbose=verbose)
        if verbose:
            print(f"After language heuristic: {len(video_metadata)} videos")

    if not budget.breached and video_metadata:
        impact_ids = _filter_by_impact(
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
            print(f"After impact filter: {len(impact_ids)} high-impact")
        by_vid = {v["id"]: v for v in video_metadata}
        video_metadata = [by_vid[i] for i in impact_ids if i in by_vid]

    if not budget.breached and video_metadata:
        video_metadata = filter_videos_by_public_health_relevance(
            video_metadata, verbose=verbose
        )
        if verbose:
            print(
                f"After semantic filter: {len(video_metadata)} public-health-relevant"
            )

    if not budget.breached:
        filtered_ids = [v["id"] for v in video_metadata]

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
        "candidates_total": len(candidate_ids) + existing_skipped,
        "candidates_new": len(candidate_ids),
        "existing_skipped": existing_skipped,
        "quota_used": budget.used,
        "quota_budget": budget.budget,
        "quota_breached": budget.breached,
    }


def handler(event, context):
    """
    AWS Lambda entrypoint for YouTube data ingestion.

    `event` may optionally provide a subset of the run_youtube_data_ingestion_pipeline
    keyword arguments; any missing keys fall back to existing defaults.
    """
    del context  # unused

    event = event or {}
    if not isinstance(event, dict):
        raise TypeError("event must be a dict or None")

    kwargs = {}
    if "search_queries" in event:
        kwargs["search_queries"] = event["search_queries"]
    if "search_order_passes" in event:
        kwargs["search_order_passes"] = list(event["search_order_passes"])
    if "exclude_existing_video_ids" in event:
        kwargs["exclude_existing_video_ids"] = bool(event["exclude_existing_video_ids"])
    if "max_search_pages" in event:
        kwargs["max_search_pages"] = int(event["max_search_pages"])
    if "min_comments_per_1k" in event:
        kwargs["min_comments_per_1k"] = float(event["min_comments_per_1k"])
    if "min_likes_per_1k" in event:
        kwargs["min_likes_per_1k"] = float(event["min_likes_per_1k"])
    if "min_views" in event:
        kwargs["min_views"] = int(event["min_views"])
    if "min_like_count" in event:
        kwargs["min_like_count"] = int(event["min_like_count"])
    if "min_comment_count" in event:
        kwargs["min_comment_count"] = int(event["min_comment_count"])
    if "percentile" in event:
        kwargs["percentile"] = float(event["percentile"])
    if "min_duration_seconds" in event:
        kwargs["min_duration_seconds"] = int(event["min_duration_seconds"])
    if "quota_budget" in event:
        kwargs["quota_budget"] = int(event["quota_budget"])
    if "verbose" in event:
        kwargs["verbose"] = bool(event["verbose"])

    result = run_youtube_data_ingestion_pipeline(**kwargs)
    return {"ok": True, **result}


if __name__ == "__main__":
    run_youtube_data_ingestion_pipeline()
