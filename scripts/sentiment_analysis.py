import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from transformers import pipeline  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env.example")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_TABLE_VIDEOS = "videos"
SUPABASE_TABLE_INSIGHTS = "insights"

# MODEL + TRANSCRIPT CLIENT
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

ytt_api = YouTubeTranscriptApi()


def clean_transcript(transcript) -> str:
    text = " ".join([snippet.text for snippet in transcript.snippets])
    text = re.sub(r"\[[^\\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", "", text)
    for pattern in [r"\b(?:um|uh|ugh|hmm)\b", r"\byou\s+know\b"]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 300) -> list:
    # Splits the text into chunks of 300 words
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


def analyze_video_sentiment(video_id: str) -> dict:
    # Fetch -> Clean -> Chunk -> Analyze -> Aggregate
    try:
        # Fetch transcript using v1.x API
        try:
            transcript = ytt_api.fetch(video_id, languages=["en"])
        except Exception:
            # If English isn't available, grab whatever language exists
            transcript = ytt_api.fetch(video_id)

        # Clean transcript
        cleaned_text = clean_transcript(transcript)

        # Chunk text
        chunks = chunk_text(cleaned_text)

        if not chunks:
            return {"error": "Transcript is empty after cleaning."}

        # Analyze sentiment for each chunk
        chunk_results = sentiment_analyzer(chunks)

        # Aggregate the results
        total_score = 0
        positive_chunks = 0
        negative_chunks = 0

        for result in chunk_results:
            # The model returns labels 'POSITIVE' or 'NEGATIVE' with a confidence score
            if result["label"] == "POSITIVE":
                positive_chunks += 1
                total_score += result["score"]  # Add confidence
            else:
                # Subtract confidence for negative chunks
                # to reflect their impact on overall sentiment
                negative_chunks += 1
                total_score -= result["score"]

        # Calculate sentiment
        avg_score = total_score / len(chunks)
        overall_sentiment = "POSITIVE" if avg_score > 0 else "NEGATIVE"

        return {
            "video_id": video_id,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(
                avg_score, 4
            ),  # Range from -1.0 (Highly Negative) to 1.0 (Highly Positive)
            "total_chunks": len(chunks),
            "positive_chunks": positive_chunks,
            "negative_chunks": negative_chunks,
            "timeline": chunk_results,  # Contains the sequential sentiment of the video
        }

    except Exception as e:
        return {"error": f"Failed to process video {video_id}: {str(e)}"}


# VIDEO ID SOURCES
def ids_from_file(filepath: str) -> list[str]:
    """Read video IDs from a text file (one ID per line, ignores blanks/comments)."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()  # strip inline comments
        if line:
            ids.append(line)
    return ids


def print_result(result: dict):
    """Print only the important stuff."""
    if "error" in result:
        print(f"  [!] {result['error']}")
        return
    score = result["sentiment_score"]
    label = result["overall_sentiment"]
    pos = result["positive_chunks"]
    neg = result["negative_chunks"]
    total = result["total_chunks"]
    print(f"  {label}  score={score:+.4f}  (pos={pos}/{total}, neg={neg}/{total})")


# ----------------------------------------------
# SUPABASE HELPERS
# ----------------------------------------------


def get_supabase_client():
    from supabase import create_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL / SUPABASE_KEY not set.")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def ids_from_supabase_without_sentiment() -> list[str]:
    """
    Pull video IDs from Supabase that do NOT already have a
    sentiment analysis row in the insights table (model = 'sentiment').
    """
    client = get_supabase_client()

    # Get all video_ids from videos table
    all_videos = client.table(SUPABASE_TABLE_VIDEOS).select("video_id").execute()
    all_ids = {r["video_id"] for r in all_videos.data}

    # Get video_ids that already have sentiment insights
    existing = (
        client.table(SUPABASE_TABLE_INSIGHTS)
        .select("video_id")
        .eq("model", "sentiment")
        .execute()
    )
    done_ids = {r["video_id"] for r in existing.data}

    remaining = list(all_ids - done_ids)
    print(
        f"  {len(all_ids)} total videos, {len(done_ids)} already analyzed, "
        f"{len(remaining)} remaining"
    )
    return remaining


def push_sentiment_to_supabase(result: dict):
    """
    Push a sentiment analysis result into the insights table.
    Maps to: video_id, model='sentiment', labels, confidence,
    claims (timeline), created_at
    """
    if "error" in result:
        return

    client = get_supabase_client()

    row = {
        "video_id": result["video_id"],
        "model": "sentiment",
        "claims": json.dumps([]),  # not applicable for sentiment
        "narratives": json.dumps([]),  # not applicable for sentiment
        "labels": json.dumps(
            {
                "overall_sentiment": result["overall_sentiment"],
                "positive_chunks": result["positive_chunks"],
                "negative_chunks": result["negative_chunks"],
                "total_chunks": result["total_chunks"],
                "timeline": result["timeline"],
            }
        ),
        "confidence": abs(result["sentiment_score"]),  # 0-1 confidence
    }

    client.table(SUPABASE_TABLE_INSIGHTS).insert(row).execute()
    print(f"    [OK] Pushed sentiment for {result['video_id']} to Supabase")


# ----------------------------------------------
# CLI
# ----------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None

    push_to_db = "--push" in sys.argv

    if mode is None:
        # Default: single test video
        video_ids = ["dQw4w9WgXcQ"]
    elif mode == "1":
        # File mode -- expects a filepath as argv[2]
        if len(sys.argv) < 3:
            print("Usage: python sentiment_analysis.py 1 <path_to_ids.txt>")
            sys.exit(1)
        video_ids = ids_from_file(sys.argv[2])
    elif mode == "2":
        # Supabase mode -- pull videos without sentiment analysis
        video_ids = ids_from_supabase_without_sentiment()
        push_to_db = True  # auto-push when pulling from supabase
    else:
        print(f"Unknown mode '{mode}'. Use: no args | 1 <file> | 2 (supabase)")
        sys.exit(1)

    if not video_ids:
        print("No videos to analyze.")
        sys.exit(0)

    print(f"Analyzing {len(video_ids)} video(s)...  (push_to_db={push_to_db})\n")

    all_results = []
    for vid in video_ids:
        print(f"[{vid}]")
        result = analyze_video_sentiment(vid)
        print_result(result)
        all_results.append(result)

        if push_to_db and "error" not in result:
            push_sentiment_to_supabase(result)

    # Summary
    successes = [r for r in all_results if "error" not in r]
    failures = len(all_results) - len(successes)
    if successes:
        avg = sum(r["sentiment_score"] for r in successes) / len(successes)
        print(f"\n{'-' * 50}")
        print(
            f"Total: {len(all_results)} videos  |"
            f"  Analyzed: {len(successes)}  |  Failed: {failures}"
        )
        print(f"Average sentiment score: {avg:+.4f}")
    else:
        print(f"\nNo videos could be analyzed ({failures} failed).")

# Usage:
#   python sentiment_analysis.py                -> test single video (no db push)
#   python sentiment_analysis.py 1 ids.txt      -> from file (add --push to save)
#   python sentiment_analysis.py 2              -> from supabase (auto-push)
