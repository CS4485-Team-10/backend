# Suppress noisy model output first (before any transformers imports)
import os
import sys

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import logging
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from transformers import pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

load_dotenv(Path(__file__).resolve().parent / ".env.example")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_TABLE_VIDEOS = "videos"

# using cardiffnlp because it gives us 3 classes (neg/neu/pos) instead of just pos/neg
# this lets us compute a real gradient: POS - NEG gives a -1 to +1 range
sentiment_analyzer = pipeline(  # type: ignore[call-overload, arg-type]
    "sentiment-analysis",  # type: ignore[arg-type]
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,  # need all 3 scores, not just the top one
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
    # 300 words per chunk works well, model cap is 512 tokens
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


def chunk_to_gradient(all_class_scores: list) -> float:
    # takes the 3-class output and collapses it to a single number
    # neutral is intentionally dropped -- we only care about the pos/neg split
    # result is in -1.0 to +1.0 range
    scores = {r["label"].lower(): r["score"] for r in all_class_scores}
    return round(scores.get("positive", 0.0) - scores.get("negative", 0.0), 4)


def analyze_video_sentiment(video_id: str) -> dict:
    # try english first, fall back to whatever's available
    try:
        try:
            transcript = ytt_api.fetch(video_id, languages=["en"])
        except Exception:
            transcript = ytt_api.fetch(video_id)

        cleaned_text = clean_transcript(transcript)
        chunks = chunk_text(cleaned_text)

        if not chunks:
            return {"error": "Transcript is empty after cleaning."}

        chunk_results = sentiment_analyzer(chunks)
        chunk_scores = [chunk_to_gradient(r) for r in chunk_results]

        positive_chunks = sum(1 for s in chunk_scores if s > 0)
        negative_chunks = sum(1 for s in chunk_scores if s <= 0)
        avg_score = round(sum(chunk_scores) / len(chunk_scores), 4)

        # anything within 0.1 of zero we just call neutral, too noisy otherwise
        if avg_score > 0.1:
            overall_sentiment = "POSITIVE"
        elif avg_score < -0.1:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"

        return {
            "video_id": video_id,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": avg_score,  # -1.0 to +1.0
            "total_chunks": len(chunks),
            "positive_chunks": positive_chunks,
            "negative_chunks": negative_chunks,
            "timeline": chunk_scores,  # gradient per chunk, in order
        }

    except Exception as e:
        return {"error": f"Failed to process video {video_id}: {str(e)}"}


# VIDEO ID SOURCES
def ids_from_file(filepath: str) -> list[str]:
    # one video ID per line, # for comments, blank lines ignored
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            ids.append(line)
    return ids


def print_result(result: dict):
    if "error" in result:
        print(f"  [!] {result['error']}")
        return
    score = result["sentiment_score"]
    label = result["overall_sentiment"]
    pos = result["positive_chunks"]
    neg = result["negative_chunks"]
    total = result["total_chunks"]
    print(f"  {label}  score={score:+.4f}  (pos={pos}/{total}, neg={neg}/{total})")


def get_supabase_client():
    from supabase import create_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL / SUPABASE_KEY not set.")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def ids_from_supabase_all_videos() -> list[str]:
    # get all video IDs from the videos table
    client = get_supabase_client()
    all_videos = client.table(SUPABASE_TABLE_VIDEOS).select("video_id").execute()
    video_ids = [str(r["video_id"]) for r in all_videos.data]  # type: ignore[index]
    print(f"  Found {len(video_ids)} video(s) in database")
    return video_ids


def save_sentiment_to_json(
    results: list[dict], output_path: str = "video_sentiment_results.json"
):
    """Save video sentiment results to JSON file for analytics."""
    if not results:
        return

    # Filter out errors
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        print("    [!] No valid results to save")
        return

    output_file = Path(output_path)
    output_file.write_text(json.dumps(valid_results, indent=2))
    print(
        f"    [OK] Saved {len(valid_results)} video sentiment result(s) to {output_path}"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    save_json = "--save" in sys.argv
    output_file = "video_sentiment_results.json"

    if mode is None:
        video_ids = ["dQw4w9WgXcQ"]
    elif mode == "1":
        if len(sys.argv) < 3:
            print("Usage: python sentiment_analysis.py 1 <path_to_ids.txt> [--save]")
            sys.exit(1)
        video_ids = ids_from_file(sys.argv[2])
    elif mode == "2":
        video_ids = ids_from_supabase_all_videos()
        save_json = True  # always save when pulling from supabase
    else:
        print(f"Unknown mode '{mode}'. Use: no args | 1 <file> [--save] | 2")
        sys.exit(1)

    if not video_ids:
        print("No videos to analyze.")
        sys.exit(0)

    print(
        f"Analyzing {len(video_ids)} video(s) for VIDEO-LEVEL sentiment...  (save_json={save_json})\n"
    )

    all_results = []
    for vid in video_ids:
        print(f"[{vid}]")
        result = analyze_video_sentiment(vid)
        print_result(result)
        all_results.append(result)

    # Summary
    successes = [r for r in all_results if "error" not in r]
    failures = len(all_results) - len(successes)
    if successes:
        avg = sum(r["sentiment_score"] for r in successes) / len(successes)
        print(f"\n{'-' * 50}")
        print(
            f"Total: {len(all_results)} videos  |  Analyzed: {len(successes)}  |  Failed: {failures}"
        )
        print(f"Average sentiment score: {avg:+.4f}")

        if save_json:
            save_sentiment_to_json(all_results, output_file)
    else:
        print(f"\nNo videos could be analyzed ({failures} failed).")

# Usage:
#   python sentiment_analysis.py                -> test single video (console output only)
#   python sentiment_analysis.py 1 ids.txt      -> from file (add --save for JSON output)
#   python sentiment_analysis.py 2              -> from supabase (auto-save to JSON)
