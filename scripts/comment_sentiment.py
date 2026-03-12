"""
YouTube Comment Sentiment
Pulls comments from YouTube videos and runs sentiment analysis on them using DistilBERT.

This is a standalone script for testing viability.
It does NOT push data to Supabase.

Requirements:
    pip install google-api-python-client transformers torch

Usage:
    python comment_sentiment.py                          -> test with default video
    python comment_sentiment.py <VIDEO_ID>               -> single video
    python comment_sentiment.py <VIDEO_ID1> <VIDEO_ID2>  -> multiple videos
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from transformers import pipeline

load_dotenv(Path(__file__).resolve().parent / ".env.example", override=True)

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# Load sentiment model (same as transcript sentiment)
print("Loading sentiment model...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
print("Model loaded.\n")


# ──────────────────────────────────────────────
# YOUTUBE COMMENTS FETCHER
# ──────────────────────────────────────────────

def fetch_comments(video_id: str, max_comments: int = 100) -> list[dict]:
    """
    Fetch top-level comments for a YouTube video using the Data API v3.
    Returns a list of dicts with 'author', 'text', 'likes', 'published_at'.
    """
    from googleapiclient.discovery import build

    if not YOUTUBE_API_KEY:
        print("Error: YOUTUBE_API_KEY not set in .env.example")
        sys.exit(1)

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                order="relevance",
                textFormat="plainText",
            )
            response = request.execute()
        except Exception as e:
            error_str = str(e)
            if "commentsDisabled" in error_str:
                print(f"  [!] Comments are disabled for video {video_id}")
            elif "videoNotFound" in error_str:
                print(f"  [!] Video {video_id} not found")
            else:
                print(f"  [!] API error for {video_id}: {error_str[:200]}")
            return comments

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": snippet.get("authorDisplayName", "Unknown"),
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# ──────────────────────────────────────────────
# SENTIMENT ANALYSIS ON COMMENTS

def analyze_comment_sentiment(comments: list[dict]) -> dict:
    """
    Run sentiment analysis on a list of comment dicts.
    Returns aggregated stats and per-comment results.
    """
    if not comments:
        return {"error": "No comments to analyze"}

    texts = [c["text"] for c in comments if c["text"].strip()]

    if not texts:
        return {"error": "All comments were empty"}

    # Truncate very long comments (model max is 512 tokens)
    texts_truncated = [t[:512] for t in texts]

    # Batch inference
    results = sentiment_analyzer(texts_truncated)

    # Attach results back to comments
    detailed = []
    for i, (comment, result) in enumerate(zip(comments, results)):
        detailed.append({
            "author": comment["author"],
            "text": comment["text"][:200],  # truncate for display
            "likes": comment["likes"],
            "sentiment": result["label"],
            "confidence": round(result["score"], 4),
        })

    # Aggregate
    positive = sum(1 for r in results if r["label"] == "POSITIVE")
    negative = sum(1 for r in results if r["label"] == "NEGATIVE")
    total = len(results)

    avg_score = sum(
        r["score"] if r["label"] == "POSITIVE" else -r["score"]
        for r in results
    ) / total

    # Top positive and negative comments (by confidence)
    sorted_positive = sorted(
        [d for d in detailed if d["sentiment"] == "POSITIVE"],
        key=lambda x: x["confidence"],
        reverse=True,
    )
    sorted_negative = sorted(
        [d for d in detailed if d["sentiment"] == "NEGATIVE"],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "total_comments": total,
        "positive_count": positive,
        "negative_count": negative,
        "positive_pct": round(positive / total * 100, 1),
        "negative_pct": round(negative / total * 100, 1),
        "avg_sentiment_score": round(avg_score, 4),
        "overall_sentiment": "POSITIVE" if avg_score > 0 else "NEGATIVE",
        "top_positive": sorted_positive[:3],
        "top_negative": sorted_negative[:3],
        "all_comments": detailed,
    }


# ──────────────────────────────────────────────
# PRETTY PRINT
# ──────────────────────────────────────────────

def print_comment_analysis(video_id: str, analysis: dict):
    print(f"\n{'=' * 60}")
    print(f"  COMMENT SENTIMENT -- {video_id}")
    print(f"{'-' * 60}")

    if "error" in analysis:
        print(f"  [!] {analysis['error']}")
        return

    label = analysis["overall_sentiment"]
    icon = "[+]" if label == "POSITIVE" else "[-]"

    print(f"  Overall: {icon} {label}  (score: {analysis['avg_sentiment_score']:+.4f})")
    print(f"  Comments: {analysis['total_comments']}  "
          f"| pos: {analysis['positive_count']} ({analysis['positive_pct']}%)  "
          f"| neg: {analysis['negative_count']} ({analysis['negative_pct']}%)")

    if analysis["top_positive"]:
        print(f"\n  Top Positive Comments:")
        for c in analysis["top_positive"]:
            print(f"    [+] [{c['confidence']:.0%}] {c['text'][:100]}...")
            print(f"       -- {c['author']} (likes: {c['likes']})")

    if analysis["top_negative"]:
        print(f"\n  Top Negative Comments:")
        for c in analysis["top_negative"]:
            print(f"    [-] [{c['confidence']:.0%}] {c['text'][:100]}...")
            print(f"       -- {c['author']} (likes: {c['likes']})")

    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Default test videos (popular health-related)
    DEFAULT_TEST_VIDEOS = [
        "dQw4w9WgXcQ",  # Rick Astley -- just for testing the pipeline works
    ]

    # Parse --max first so we can exclude it from video IDs
    max_comments = 50  # Keep it small for viability test
    max_idx = -1
    if "--max" in sys.argv:
        max_idx = sys.argv.index("--max")
        if max_idx + 1 < len(sys.argv):
            max_comments = int(sys.argv[max_idx + 1])

    if len(sys.argv) > 1:
        skip = {max_idx, max_idx + 1} if max_idx >= 0 else set()
        video_ids = [
            v for i, v in enumerate(sys.argv)
            if i > 0 and i not in skip and not v.startswith("--")
        ]
    else:
        video_ids = DEFAULT_TEST_VIDEOS

    if not video_ids:
        video_ids = DEFAULT_TEST_VIDEOS

    print(f"Fetching up to {max_comments} comments per video for {len(video_ids)} video(s)...\n")

    all_analyses = {}

    for vid in video_ids:
        print(f"[{vid}] Fetching comments...")
        comments = fetch_comments(vid, max_comments=max_comments)
        print(f"  Got {len(comments)} comments")

        if comments:
            print(f"  Running sentiment analysis...")
            analysis = analyze_comment_sentiment(comments)
            all_analyses[vid] = analysis
            print_comment_analysis(vid, analysis)
        else:
            all_analyses[vid] = {"error": "No comments fetched"}
            print(f"  [!] No comments to analyze\n")

    # Save raw results to JSON for inspection
    output_path = Path(__file__).resolve().parent / "comment_sentiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_analyses, f, indent=2, default=str)
    print(f"Full results saved to: {output_path}")

# Usage:
#   python comment_sentiment.py                                -> default test video
#   python comment_sentiment.py VIDEO_ID1 VIDEO_ID2            -> specific videos
#   python comment_sentiment.py VIDEO_ID --max 200             -> more comments
