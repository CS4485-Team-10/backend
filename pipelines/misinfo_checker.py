# Misinformation checks for YouTube transcripts.
import os
import sys

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from transformers import pipeline
from transformers import pipeline as hf_pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

load_dotenv(Path(__file__).resolve().parent / ".env")

GOOGLE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_TABLE_VIDEOS = "videos"
SUPABASE_TABLE_CLAIMS = "claims"

FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

nli_model = pipeline(  # type: ignore[call-overload]
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

ytt_api = YouTubeTranscriptApi()

MISINFO_PATTERNS = [
    # Vaccines
    (r"vaccines?\s+caus(e|es|ed|ing)\s+autism", "Vaccines cause autism", "high"),
    (
        r"vaccines?\s+(are|is)\s+(poison|toxic|dangerous)",
        "Vaccines are toxic/dangerous",
        "high",
    ),
    (
        r"vaccines?\s+(contain|have)\s+microchips?",
        "Vaccines contain microchips",
        "high",
    ),
    (
        r"vaccines?\s+(alter|change|modify)\s+(your\s+)?dna",
        "Vaccines alter DNA",
        "high",
    ),
    (
        r"(don.?t|never|stop)\s+vaccinate?\s+(your\s+)?(kids?|children|babies?)",
        "Anti-vaccination rhetoric",
        "high",
    ),
    (
        r"natural\s+immunity\s+is\s+(all|enough|better|superior)",
        "Natural immunity superiority claims",
        "medium",
    ),
    # COVID-specific
    (r"covid.{0,10}(hoax|fake|planned|plandemic)", "COVID is a hoax/planned", "high"),
    (
        r"5g\s+(caus|spread|transmit).{0,20}(covid|virus|disease)",
        "5G causes COVID",
        "high",
    ),
    (r"ivermectin\s+(cure|treat|heal).{0,20}covid", "Ivermectin cures COVID", "high"),
    (
        r"hydroxychloroquine\s+(cure|treat|heal).{0,20}covid",
        "Hydroxychloroquine cures COVID",
        "high",
    ),
    (r"masks?\s+(don.?t|do\s+not|never)\s+work", "Masks don't work", "medium"),
    # Big Pharma / conspiracy
    (
        r"big\s+pharma\s+(is\s+)?(hiding|suppressing|covering)",
        "Big Pharma suppression conspiracy",
        "medium",
    ),
    (
        r"cure\s+for\s+cancer\s+(is\s+)?(being\s+)?(hidden|suppressed|covered)",
        "Cancer cure is hidden",
        "high",
    ),
    (
        r"doctors?\s+(are|is)\s+(all\s+)?(lying|paid\s+off|corrupt|bribed)",
        "Doctors are corrupt/lying",
        "medium",
    ),
    (
        r"(pharma|drug)\s+companies?\s+(don.?t|do\s+not)\s+want\s+you\s+to\s+know",
        "Pharma hiding information",
        "medium",
    ),
    (
        r"(government|fda|cdc|who)\s+(is\s+)?(lying|corrupt|hiding|cover)",
        "Government health conspiracy",
        "medium",
    ),
    # Alternative medicine overclaims
    (
        r"(essential\s+oils?|herbs?|crystals?|homeopathy)\s+(cure|heal|treat).{0,20}(cancer|diabetes|hiv|aids)",
        "Alt medicine cures serious disease",
        "high",
    ),
    (
        r"(alkaline|detox|cleanse)\s+(cure|heal|prevent).{0,20}(cancer|disease)",
        "Alkaline/detox cures disease",
        "high",
    ),
    (
        r"(coffee\s+enema|turpentine|bleach|mms)\s+(cure|heal|treat)",
        "Dangerous substance as medicine",
        "high",
    ),
    # Anti-science
    (
        r"(germ\s+theory|evolution)\s+(is\s+)?(a\s+)?(lie|myth|fraud|hoax|wrong)",
        "Germ theory denial",
        "high",
    ),
    (
        r"(fluoride|water)\s+(is\s+)?(poison|mind\s+control|toxic)",
        "Fluoride conspiracy",
        "medium",
    ),
    (
        r"(chemtrails?)\s+(are|is)\s+(spraying|poison|real)",
        "Chemtrails conspiracy",
        "medium",
    ),
]

COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE), desc, sev) for p, desc, sev in MISINFO_PATTERNS
]

HEALTH_CLAIM_TYPES = [
    "a specific health treatment or cure",
    "a claim about vaccine safety or efficacy",
    "a claim about a drug or medication",
    "a dietary or nutritional health claim",
    "a claim about a disease cause or mechanism",
    "a conspiracy theory about healthcare",
    "a claim about government or institutional health policy",
    "general health education or information",
]


@dataclass
class PatternMatch:
    pattern_description: str
    severity: str
    match_count: int


@dataclass
class FactCheckResult:
    claim_text: str
    claimant: str | None
    rating: str
    publisher: str
    url: str


@dataclass
class ClaimAnalysis:
    claim_text: str
    claim_type: str
    claim_type_confidence: float
    entailment_label: str  # "supported", "refuted", "neutral"
    entailment_confidence: float
    fact_checks: list[FactCheckResult] = field(default_factory=list)


@dataclass
class VideoMisinfoReport:
    video_id: str
    error: str | None = None
    transcript_length_words: int = 0
    pattern_matches: list[PatternMatch] = field(default_factory=list)
    high_severity_count: int = 0
    medium_severity_count: int = 0
    claims_analyzed: list[ClaimAnalysis] = field(default_factory=list)
    risk_level: str = "low"
    risk_reasons: list[str] = field(default_factory=list)


def fetch_transcript(video_id: str) -> str | None:
    try:
        client = get_supabase_client()
        result = (
            client.table("transcripts")
            .select("cleaned_transcript_txt")
            .eq("video_id", video_id)
            .execute()
        )

        if result.data and len(result.data) > 0:
            text = result.data[0]["cleaned_transcript_txt"]
        else:
            try:
                transcript = ytt_api.fetch(video_id, languages=["en"])
            except Exception:
                transcript = ytt_api.fetch(video_id)

            text = " ".join(s.text for s in transcript.snippets)
        text = re.sub(r"\[[^\]]*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\b(?:um|uh|ugh|hmm)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return None


def extract_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw:
        s = s.strip()
        word_count = len(s.split())
        if 5 <= word_count <= 60:
            sentences.append(s)
    return sentences


def scan_patterns(text: str) -> list[PatternMatch]:
    matches = []
    for pattern, description, severity in COMPILED_PATTERNS:
        found = pattern.findall(text)
        if found:
            matches.append(
                PatternMatch(
                    pattern_description=description,
                    severity=severity,
                    match_count=len(found),
                )
            )
    return matches


def search_fact_checks(query: str, max_results: int = 3) -> list[FactCheckResult]:
    if not GOOGLE_API_KEY:
        return []

    try:
        resp = requests.get(
            FACT_CHECK_API_URL,
            params={
                "key": GOOGLE_API_KEY,
                "query": query[:200],
                "languageCode": "en",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        results = []
        for claim in data.get("claims", [])[:max_results]:
            for review in claim.get("claimReview", []):
                results.append(
                    FactCheckResult(
                        claim_text=claim.get("text", ""),
                        claimant=claim.get("claimant"),
                        rating=review.get("textualRating", "Unknown"),
                        publisher=review.get("publisher", {}).get("name", "Unknown"),
                        url=review.get("url", ""),
                    )
                )
        return results
    except Exception:
        return []


def classify_claim_type(sentence: str) -> tuple[str, float]:
    result = nli_model(sentence, candidate_labels=HEALTH_CLAIM_TYPES)
    return result["labels"][0], round(result["scores"][0], 4)


def check_entailment(claim: str, evidence: str | None = None) -> tuple[str, float]:
    consensus_hypotheses = [
        "This claim is consistent with established medical science.",
        "This claim contradicts established medical science.",
        "This claim is an opinion or cannot be verified.",
    ]
    label_map = {
        consensus_hypotheses[0]: "supported",
        consensus_hypotheses[1]: "refuted",
        consensus_hypotheses[2]: "neutral",
    }
    result = nli_model(claim, candidate_labels=consensus_hypotheses)
    top_label = result["labels"][0]
    return label_map[top_label], round(result["scores"][0], 4)


def is_health_claim(sentence: str) -> bool:
    result = nli_model(
        sentence,
        candidate_labels=[
            "a factual health or medical claim",
            "general narration or filler",
        ],
    )
    return (
        result["labels"][0] == "a factual health or medical claim"
        and result["scores"][0] > 0.6
    )


def analyze_claims(text: str, max_claims: int = 10) -> list[ClaimAnalysis]:
    sentences = extract_sentences(text)

    claim_sentences = []
    for s in sentences:
        if len(claim_sentences) >= max_claims * 3:
            break
        if is_health_claim(s):
            claim_sentences.append(s)

    analyses = []
    for sentence in claim_sentences[:max_claims]:
        claim_type, type_conf = classify_claim_type(sentence)
        entailment, ent_conf = check_entailment(sentence)

        if (
            claim_type == "general health education or information"
            and entailment == "supported"
        ):
            continue

        fact_checks = search_fact_checks(sentence)

        analyses.append(
            ClaimAnalysis(
                claim_text=sentence,
                claim_type=claim_type,
                claim_type_confidence=type_conf,
                entailment_label=entailment,
                entailment_confidence=ent_conf,
                fact_checks=fact_checks,
            )
        )

    return analyses


def check_video(video_id: str) -> VideoMisinfoReport:
    report = VideoMisinfoReport(video_id=video_id)

    text = fetch_transcript(video_id)
    if not text:
        report.error = "Could not fetch transcript"
        return report

    report.transcript_length_words = len(text.split())

    report.pattern_matches = scan_patterns(text)
    report.high_severity_count = sum(
        1 for m in report.pattern_matches if m.severity == "high"
    )
    report.medium_severity_count = sum(
        1 for m in report.pattern_matches if m.severity == "medium"
    )

    report.claims_analyzed = analyze_claims(text)

    refuted_claims = sum(
        1 for c in report.claims_analyzed if c.entailment_label == "refuted"
    )
    fact_checked_false = sum(
        1
        for c in report.claims_analyzed
        for fc in c.fact_checks
        if any(
            word in fc.rating.lower()
            for word in ["false", "pants on fire", "incorrect", "misleading", "wrong"]
        )
    )

    reasons = []
    if report.high_severity_count > 0:
        reasons.append(f"{report.high_severity_count} high-severity misinfo pattern(s)")
    if refuted_claims > 0:
        reasons.append(
            f"{refuted_claims} claim(s) flagged as contradicting medical consensus"
        )
    if fact_checked_false > 0:
        reasons.append(f"{fact_checked_false} claim(s) rated false by fact-checkers")
    if report.medium_severity_count >= 3:
        reasons.append(
            f"{report.medium_severity_count} medium-severity suspicious patterns"
        )

    report.risk_reasons = reasons

    if report.high_severity_count > 0 or fact_checked_false > 0 or refuted_claims >= 2:
        report.risk_level = "high"
    elif report.medium_severity_count > 0 or refuted_claims >= 1:
        report.risk_level = "medium"
    else:
        report.risk_level = "low"

    return report


def ids_from_file(filepath: str) -> list[str]:
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


def get_supabase_client():
    from supabase import create_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL / SUPABASE_KEY not set.")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


_claim_sentiment_pipeline = None


def _get_claim_sentiment_pipeline():
    global _claim_sentiment_pipeline
    if _claim_sentiment_pipeline is None:
        print("Loading sentiment model for claims...")
        _claim_sentiment_pipeline = hf_pipeline(  # type: ignore[call-overload]
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )
    return _claim_sentiment_pipeline


def _run_claim_sentiment(claim_text: str) -> tuple[str, float]:
    pipe = _get_claim_sentiment_pipeline()
    result = pipe(claim_text[:512])[0]
    scores = {r["label"].lower(): r["score"] for r in result}
    gradient = round(scores.get("positive", 0.0) - scores.get("negative", 0.0), 4)
    if gradient > 0.1:
        label = "POSITIVE"
    elif gradient < -0.1:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return label, gradient


def _run_fact_check_status(claim_text: str) -> str:
    results = search_fact_checks(claim_text, max_results=3)
    if not results:
        return "pending"

    FALSE_WORDS = {
        "false",
        "pants on fire",
        "incorrect",
        "misleading",
        "wrong",
        "inaccurate",
        "fake",
    }
    TRUE_WORDS = {"true", "correct", "accurate", "verified", "mostly true"}
    UNVERIFIABLE_WORDS = {"unverifiable", "unproven", "unclear", "mixed"}

    for fc in results:
        rating = fc.rating.lower()
        if any(w in rating for w in FALSE_WORDS):
            return "verified_false"
        if any(w in rating for w in TRUE_WORDS):
            return "verified_true"
        if any(w in rating for w in UNVERIFIABLE_WORDS):
            return "unverifiable"

    return "unverifiable"


def _calculate_fact_check_confidence(
    entailment_confidence: float, has_fact_checks: bool
) -> str:
    if entailment_confidence >= 0.75 and has_fact_checks:
        return "high"
    if entailment_confidence >= 0.5 or has_fact_checks:
        return "medium"
    return "low"


def _calculate_claim_risk_level(
    fact_check_status: str,
    fact_check_confidence: str,
    entailment_label: str,
    entailment_confidence: float,
) -> str:
    if fact_check_status == "verified_false":
        return "high"
    if entailment_label == "refuted" and entailment_confidence >= 0.75:
        return "high"
    if (
        fact_check_status in {"unverifiable", "pending"}
        or fact_check_confidence == "low"
        or entailment_label == "refuted"
    ):
        return "medium"
    return "low"


def process_claims_table(batch_size: int = 50):
    client = get_supabase_client()

    resp = (
        client.table("claims")
        .select(
            "claim_id, claim_text, sentiment_label, sentiment_score, "
            "fact_check_status, fact_check_confidence, risk_level"
        )
        .or_(
            "sentiment_label.is.null,"
            "sentiment_score.is.null,"
            "fact_check_status.is.null,"
            "fact_check_confidence.is.null,"
            "risk_level.is.null"
        )
        .limit(batch_size)
        .execute()
    )

    rows = resp.data
    if not rows:
        print("No unprocessed claims found.")
        return

    print(f"Processing {len(rows)} unprocessed claim(s)...\n")

    for i, row in enumerate(rows, 1):
        claim_id = row["claim_id"]
        claim_text = row["claim_text"]

        print(f"  [{i}/{len(rows)}] claim_id={claim_id}")
        print(f"    text: {claim_text[:100]}...")

        updates = {}
        if row["sentiment_label"] is None or row["sentiment_score"] is None:
            sent_label, sent_score = _run_claim_sentiment(claim_text)
            updates["sentiment_label"] = sent_label
            updates["sentiment_score"] = sent_score
            print(f"    sentiment: {sent_label} ({sent_score:+.4f})")

        if row["fact_check_status"] is None:
            fc_status = _run_fact_check_status(claim_text)
            updates["fact_check_status"] = fc_status
            print(f"    fact_check_status: {fc_status}")

        entailment_label = "neutral"
        entailment_conf = 0.0
        if row["fact_check_confidence"] is None or row["risk_level"] is None:
            entailment_label, entailment_conf = check_entailment(claim_text)

        if row["fact_check_confidence"] is None:
            fact_checks = search_fact_checks(claim_text, max_results=3)
            confidence = _calculate_fact_check_confidence(
                entailment_conf, len(fact_checks) > 0
            )
            updates["fact_check_confidence"] = confidence
            print(f"    fact_check_confidence: {confidence}")

        if row["risk_level"] is None:
            effective_status = updates.get(
                "fact_check_status", row["fact_check_status"]
            )
            effective_conf = updates.get(
                "fact_check_confidence", row["fact_check_confidence"]
            )
            if effective_status is None:
                effective_status = _run_fact_check_status(claim_text)
                updates["fact_check_status"] = effective_status
            if effective_conf is None:
                fact_checks = search_fact_checks(claim_text, max_results=3)
                effective_conf = _calculate_fact_check_confidence(
                    entailment_conf, len(fact_checks) > 0
                )
                updates["fact_check_confidence"] = effective_conf
            risk_level = _calculate_claim_risk_level(
                fact_check_status=effective_status,
                fact_check_confidence=effective_conf,
                entailment_label=entailment_label,
                entailment_confidence=entailment_conf,
            )
            updates["risk_level"] = risk_level
            print(f"    risk_level: {risk_level}")

        if updates:
            client.table("claims").update(updates).eq("claim_id", claim_id).execute()
            print(f"    [OK] Updated claim {claim_id}\n")

    print(f"Done. {len(rows)} claim(s) processed.")


def ids_from_supabase() -> list[str]:
    client = get_supabase_client()
    rows = client.table(SUPABASE_TABLE_VIDEOS).select("video_id").execute()
    return [r["video_id"] for r in rows.data]


def ids_from_supabase_with_incomplete_claims(limit: int = 1000) -> list[str]:
    """Find videos that have claims with any NULL enrichment fields."""
    client = get_supabase_client()
    rows = (
        client.table(SUPABASE_TABLE_CLAIMS)
        .select("video_id")
        .or_(
            "sentiment_label.is.null,"
            "sentiment_score.is.null,"
            "fact_check_status.is.null,"
            "fact_check_confidence.is.null,"
            "risk_level.is.null"
        )
        .limit(limit)
        .execute()
    )
    video_ids = sorted({str(r["video_id"]) for r in rows.data})
    print(f"  Found {len(video_ids)} video(s) with incomplete claim enrichment")
    return video_ids


def ids_from_supabase_without_factcheck() -> list[str]:
    """
    Find videos that have claims but those claims haven't been
    fact-checked yet (fact_check_status is NULL). This is more targeted
    than ids_from_supabase_with_incomplete_claims() which checks all fields.
    """
    client = get_supabase_client()

    # Get all videos that have claims
    all_claims = client.table(SUPABASE_TABLE_CLAIMS).select("video_id").execute()
    videos_with_claims = set(r["video_id"] for r in all_claims.data)

    # Get videos where claims have been fact-checked (fact_check_status is NOT NULL)
    checked_claims = (
        client.table(SUPABASE_TABLE_CLAIMS)
        .select("video_id")
        .not_.is_("fact_check_status", "null")
        .execute()
    )
    videos_already_checked = set(r["video_id"] for r in checked_claims.data)

    # Videos needing fact-check = have claims but none are checked
    remaining = sorted(list(videos_with_claims - videos_already_checked))
    print(
        f"  {len(videos_with_claims)} videos with claims, "
        f"{len(videos_already_checked)} already fact-checked, "
        f"{len(remaining)} remaining"
    )
    return remaining


RISK_ICONS = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}


def print_report(report: VideoMisinfoReport):
    icon = RISK_ICONS.get(report.risk_level, "[?]")

    if report.error:
        print(f"  [!] {report.error}\n")
        return

    print(
        f"  Risk: {icon} {report.risk_level.upper()}  ({report.transcript_length_words} words)"
    )

    # Pattern matches
    if report.pattern_matches:
        print("  Pattern flags:")
        for m in report.pattern_matches:
            sev_icon = "[HIGH]" if m.severity == "high" else "[MED]"
            print(f"    {sev_icon} {m.pattern_description} (x{m.match_count})")
    else:
        print("  Pattern flags: None")

    # Claims
    if report.claims_analyzed:
        print(f"  Claims analyzed: {len(report.claims_analyzed)}")
        for i, c in enumerate(report.claims_analyzed, 1):
            ent_icon = {"supported": "[OK]", "refuted": "[X]", "neutral": "[-]"}.get(
                c.entailment_label, "?"
            )
            print(
                f"    {i}. {ent_icon} [{c.entailment_label}] ({c.entailment_confidence:.0%}) {c.claim_text[:120]}"
            )
            if c.fact_checks:
                for fc in c.fact_checks:
                    print(f'       [FC] {fc.publisher}: "{fc.rating}"')
                    print(f"            {fc.url}")
    else:
        print("  Claims: No notable health claims detected")

    # Risk reasons
    if report.risk_reasons:
        print("  Reasons:")
        for r in report.risk_reasons:
            print(f"    -> {r}")

    print()


def run_misinfo_videos_pipeline(
    video_ids: list[str],
    *,
    write_json: bool = False,
    json_path: str = "misinfo_report.json",
) -> dict:
    """Run misinformation checks for a list of video IDs."""
    if not video_ids:
        print("No videos to check.")
        return {
            "ok": True,
            "video_ids": [],
            "total": 0,
            "checked": 0,
            "failed": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "total_patterns": 0,
            "total_claims": 0,
            "saved_json": False,
            "json_path": None,
            "reports": [],
        }

    print(f"Checking {len(video_ids)} video(s) for misinformation...\n")

    all_reports: list[VideoMisinfoReport] = []
    for i, vid in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}] {vid}")
        report = check_video(vid)
        print_report(report)
        all_reports.append(report)

    checked = [r for r in all_reports if r.error is None]
    failed = len(all_reports) - len(checked)
    high = sum(1 for r in checked if r.risk_level == "high")
    medium = sum(1 for r in checked if r.risk_level == "medium")
    low = sum(1 for r in checked if r.risk_level == "low")
    total_patterns = sum(len(r.pattern_matches) for r in checked)
    total_claims = sum(len(r.claims_analyzed) for r in checked)

    print(f"{'=' * 60}")
    print("MISINFORMATION CHECK SUMMARY")
    print(f"{'-' * 60}")
    print(
        f"  Videos checked:     {len(checked)}/{len(all_reports)}  (failed: {failed})"
    )
    print(
        f"  Risk breakdown:     [HIGH] {high} high  |  [MED] {medium} medium  |  [LOW] {low} low"
    )
    print(f"  Pattern flags:      {total_patterns} total across all videos")
    print(f"  Claims analyzed:    {total_claims} total")
    if high > 0:
        print(f"\n  [!] {high} video(s) flagged HIGH RISK -- review recommended")
    print(f"{'=' * 60}")

    saved_json = False
    if write_json:
        output = [asdict(r) for r in all_reports]
        path = Path(json_path)
        path.write_text(json.dumps(output, indent=2, default=str))
        print(f"\nFull report saved to {path}")
        saved_json = True

    return {
        "ok": True,
        "video_ids": [r.video_id for r in all_reports],
        "total": len(all_reports),
        "checked": len(checked),
        "failed": failed,
        "high": high,
        "medium": medium,
        "low": low,
        "total_patterns": total_patterns,
        "total_claims": total_claims,
        "saved_json": saved_json,
        "json_path": json_path if saved_json else None,
        "reports": [asdict(r) for r in all_reports],
    }


def handler(event, context):
    """
    AWS Lambda entrypoint for misinformation checks.

    `event` may provide:
      - action: "videos" (default) or "claims_batch"
      - video_ids: explicit list of IDs (for action == "videos")
      - mode: None | "1" | "2" | "3" (mirrors CLI modes) for videos
      - ids_file: path used with mode "1"
      - write_json: bool
      - json_path: output path for JSON
      - batch_size: int (for action == "claims_batch")
    """
    del context  # unused

    event = event or {}
    if not isinstance(event, dict):
        raise TypeError("event must be a dict or None")

    action = event.get("action", "videos")

    if action == "claims_batch":
        batch_size = int(event.get("batch_size", 50))
        process_claims_table(batch_size=batch_size)
        return {"ok": True, "action": "claims_batch", "batch_size": batch_size}

    write_json = bool(event.get("write_json", False))
    json_path = event.get("json_path", "misinfo_report.json")

    if "video_ids" in event and event["video_ids"]:
        video_ids = list(event["video_ids"])
    else:
        mode = event.get("mode")

        if mode is None:
            video_ids = ["dQw4w9WgXcQ", "T9itjMTqQ8Q"]
        elif mode == "1":
            ids_file = event.get("ids_file")
            if not ids_file:
                raise ValueError("ids_file is required when mode == '1'")
            video_ids = ids_from_file(ids_file)
        elif mode == "2":
            video_ids = ids_from_supabase()
        elif mode == "3":
            video_ids = ids_from_supabase_with_incomplete_claims()
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use: None | '1' with ids_file | '2' | '3'"
            )

    return run_misinfo_videos_pipeline(
        video_ids, write_json=write_json, json_path=json_path
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "4":
        batch = 50
        if "--batch" in sys.argv:
            idx = sys.argv.index("--batch")
            if idx + 1 < len(sys.argv):
                batch = int(sys.argv[idx + 1])
        process_claims_table(batch_size=batch)
        sys.exit(0)

    write_json = "--json" in sys.argv

    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 >= len(sys.argv):
            print("Usage: python misinfo_checker.py --video <VIDEO_ID>")
            sys.exit(1)
        video_ids = [sys.argv[idx + 1]]
    else:
        mode = sys.argv[1] if len(sys.argv) > 1 else None

        if mode is None:
            video_ids = ["dQw4w9WgXcQ", "T9itjMTqQ8Q"]
        elif mode == "1":
            if len(sys.argv) < 3:
                print("Usage: python misinfo_checker.py 1 <path_to_ids.txt>")
                sys.exit(1)
            video_ids = ids_from_file(sys.argv[2])
        elif mode == "2":
            video_ids = ids_from_supabase()
        elif mode == "3":
            video_ids = ids_from_supabase_without_factcheck()
        else:
            print(
                f"Unknown mode '{mode}'. Use: no args | 1 <file> | 2 | 3 | --video <ID>"
            )
            sys.exit(1)

    run_misinfo_videos_pipeline(
        video_ids, write_json=write_json, json_path="misinfo_report.json"
    )
