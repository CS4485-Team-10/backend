"""Extract claims, match narratives, and persist first-pass insights."""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from supabase import create_client

from pipelines.narrative_matching import (
    MatchDecision,
    NarrativeCandidate,
    build_candidate_pool,
    get_embedder_from_env,
    match_claim_to_narratives,
    refresh_pool_with_new,
)
from pipelines.shared import LLMProvider

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

_PAGE_SIZE = 1000

# Defensive max lengths for parsed LLM fields (downstream of JSON only; prompts unchanged).
_MAX_CLAIM_TEXT_CHARS = 100_000
_MAX_NARRATIVE_THEME_CHARS = 500
_MAX_NARRATIVE_CATEGORY_CHARS = 200
_MAX_NARRATIVE_DESCRIPTION_CHARS = 4_000
_MAX_NARRATIVE_DETAILS_CHARS = 8_000

_DEFAULT_NARRATIVE_CATEGORY = "Uncategorized"


def _clip_str(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len]


def _normalize_optional_str(raw: Any, max_len: int) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    return _clip_str(s, max_len)


def _normalize_confidence(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_parsed_claim(raw: Any) -> Optional[Dict[str, Any]]:
    """Return a canonical claim dict, or None."""
    if not isinstance(raw, dict):
        log.debug("Skipping non-dict claim entry: %s", type(raw).__name__)
        return None

    text_raw = raw.get("text")
    if text_raw is None:
        text_raw = ""
    elif not isinstance(text_raw, str):
        text_raw = str(text_raw)
    text = _clip_str(text_raw.strip(), _MAX_CLAIM_TEXT_CHARS)
    if not text:
        return None

    category = _normalize_optional_str(
        raw.get("narrative_category"), _MAX_NARRATIVE_CATEGORY_CHARS
    )
    if not category:
        category = _DEFAULT_NARRATIVE_CATEGORY

    return {
        "text": text,
        "confidence": _normalize_confidence(raw.get("confidence")),
        "narrative_theme": _normalize_optional_str(
            raw.get("narrative_theme"), _MAX_NARRATIVE_THEME_CHARS
        ),
        "narrative_category": category,
        "narrative_description": _normalize_optional_str(
            raw.get("narrative_description"), _MAX_NARRATIVE_DESCRIPTION_CHARS
        ),
        "narrative_details": _normalize_optional_str(
            raw.get("narrative_details"), _MAX_NARRATIVE_DETAILS_CHARS
        ),
    }


def _get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# 1. Supabase reads
# ---------------------------------------------------------------------------


def _fetch_transcripts_without_claims(sb) -> List[Dict[str, Any]]:
    """Return transcript rows with no matching claims."""
    all_transcripts: List[Dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            sb.table("transcripts")
            .select("video_id, transcript_id, cleaned_transcript_txt")
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        all_transcripts.extend(page.data)
        if len(page.data) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    if not all_transcripts:
        return []

    claimed_ids: set[str] = set()
    offset = 0
    while True:
        page = (
            sb.table("claims")
            .select("transcript_id")
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        for row in page.data:
            claimed_ids.add(str(row["transcript_id"]))
        if len(page.data) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    return [t for t in all_transcripts if str(t["transcript_id"]) not in claimed_ids]


def _fetch_all_narratives(sb) -> List[Dict[str, Any]]:
    """Return all narrative rows via pagination."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            sb.table("narratives")
            .select(
                "narrative_id, narrative_label, narrative_risk_score, "
                "narrative_category, narrative_description, narrative_details"
            )
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        rows.extend(page.data)
        if len(page.data) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# 2. Replace existing transcript claims
# ---------------------------------------------------------------------------


def _delete_existing_insights(sb, transcript_id: str) -> None:
    """Delete old bridge rows and claims for a transcript."""
    existing_claims = (
        sb.table("claims")
        .select("claim_id")
        .eq("transcript_id", transcript_id)
        .execute()
    )
    claim_ids = [r["claim_id"] for r in existing_claims.data]
    if not claim_ids:
        return

    for cid in claim_ids:
        sb.table("claim_narratives").delete().eq("claim_id", cid).execute()
    sb.table("claims").delete().eq("transcript_id", transcript_id).execute()


# ---------------------------------------------------------------------------
# 3. Chunking & LLM extraction
# ---------------------------------------------------------------------------


def _chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    """Split transcript text into model-sized chunks."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current_len + para_len > max_chars and buffer:
            chunks.append("\n\n".join(buffer))
            buffer = [para]
            current_len = len(para)
        else:
            buffer.append(para)
            current_len += para_len

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks if chunks else [text]


_SYSTEM = """You extract GENERALIZABLE HEALTH-RELATED CLAIMS from video transcripts.

Your job is to identify only claims that are useful for health insight analysis, misinformation detection, or narrative grouping.
For each valid claim, also generate reusable narrative metadata:
`narrative_theme`, `narrative_category`, `narrative_description`, and `narrative_details`.

INCLUDE ONLY:
- Claims about health, disease, symptoms, diagnosis, treatment, prevention, medication, side effects, risk factors, sleep, mental health, biology, or healthcare systems
- Claims that are generalizable beyond one individual's private life
- Claims that a reviewer could plausibly verify, fact-check, or group with similar health claims
- Claims stated as fact, or clearly implied as fact

EXCLUDE:
- Personal life updates, autobiographical details, or emotional experiences without broader health relevance
- Motivational, inspirational, spiritual, or self-help statements
- Vague opinions, feelings, or reflections
- Social conflict or interpersonal drama unless it directly supports a broader health-related claim
- Redundant restatements of the same idea
- Purely historical, geographic, or non-health content

SPECIAL RULE FOR PERSONAL EXPERIENCES:
If a speaker describes a personal experience, only extract a claim if the statement conveys a broader health-related assertion that could apply beyond that one individual.
Example:
- KEEP: "Stimulants can help people with ADHD feel normal and focused."
- DROP: "The speaker has not seen their mom in a year and a half."

If a transcript contains no generalizable health-related claims, return:
{"claims": []}

Return ONLY valid JSON.
"""

_SCHEMA_HINT = """
{
  "claims": [
    {
      "text": "A concise, generalizable health-related claim (not a personal anecdote or motivational statement)",
      "confidence": 0.0,
      "narrative_theme": "A short, reusable health topic label (e.g., Sleep Deprivation Risks, ADHD and Substance Use, Vaccine Skepticism)",
      "narrative_category": "A coarse health category (e.g., Sleep, Mental Health, Vaccines, Chronic Disease, Healthcare Systems)",
      "narrative_description": "A concise 1-2 sentence summary of the overarching health narrative",
      "narrative_details": "A slightly richer explanation describing the kinds of claims, mechanisms, risks, or themes that belong under this narrative"
    }
  ]
}

Confidence properties:
- "confidence" must be a float between 0.0 and 1.0
- 1.0 = very confident this is a clear, valid, well-formed claim
- 0.0 = very low confidence or ambiguous claim

""".strip()


def _build_user_prompt(transcript_chunk: str) -> str:
    return f"""Extract only the GENERALIZABLE HEALTH-RELATED CLAIMS from the transcript below.

Return ONLY valid JSON in this format:
{_SCHEMA_HINT}

Rules:
- Keep only claims that are health-related and useful beyond one person's private situation
- Do NOT extract personal anecdotes unless they express a broader health claim
- Do NOT extract motivational, spiritual, or vague self-help statements
- Do NOT extract duplicate or near-duplicate claims
- Prefer concise, normalized wording over dramatic or conversational phrasing

Rules for generating narrative metadata:
- "narrative_theme" must be a short topic label, not a sentence
- Make "narrative_theme" reusable across multiple similar claims
- Focus on the underlying health topic, not the speaker, tone, or sequence
- Avoid generic labels like "Personal History", "Initial Effects", "Long-term Effects", "Encouragement"
- Avoid overly narrow labels that only fit one claim

- "narrative_category" must be a coarse reusable health bucket such as:
  Sleep, Mental Health, Vaccines, Chronic Disease, Healthcare Systems, Nutrition, Medications, Public Health, Addiction, Endocrine Health
- Use "Uncategorized" only if no reasonable category fits

- "narrative_description" should be a concise 1-2 sentence summary of the broader health narrative
- "narrative_details" should be a slightly richer explanation of the types of claims, mechanisms, risks, or health ideas that belong under that narrative
- Do not make description/details speaker-specific
- Do not make description/details motivational or vague
- Keep both description and details generalizable and health-relevant

Good examples of narrative_theme:
- Sleep Deprivation Risks
- ADHD and Substance Use
- Vaccine Skepticism
- Graves Disease Causes and Triggers
- Teen Sleep and School Start Times
- Mental Health Overprescription

Bad examples of narrative_theme:
- My Journey
- Initial Effects
- Important Truth
- Encouragement
- Healing

If there are no valid generalizable health-related claims, return:
{{"claims": []}}

Transcript:
---
{transcript_chunk}
---
"""


def _validate_json_output(text: str) -> Dict[str, Any]:
    """Parse and validate JSON output."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object (dict)")
    if "claims" not in parsed:
        raise ValueError("Model output must contain a 'claims' key")
    if not isinstance(parsed["claims"], list):
        raise ValueError("'claims' must be a list")
    return parsed


def _extract_claims(
    transcript_text: str,
    provider: LLMProvider,
    *,
    max_chars: int = 12000,
    retries: int = 2,
) -> List[Dict[str, Any]]:
    """Chunk transcript, call LLM, normalize, dedupe, return claims."""
    chunks = _chunk_text(transcript_text, max_chars=max_chars)
    raw_items: List[Any] = []

    for chunk in chunks:
        user_prompt = _build_user_prompt(chunk)
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                raw = provider.generate_response(
                    system=_SYSTEM, user_prompt=user_prompt
                )
                parsed = _validate_json_output(raw)
                raw_items.extend(parsed.get("claims", []))
                break
            except Exception as e:
                last_error = e
                if attempt == retries:
                    raise last_error from last_error

    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for item in raw_items:
        norm = _normalize_parsed_claim(item)
        if norm is None:
            continue
        key = norm["text"]
        if key not in seen:
            seen.add(key)
            unique.append(norm)
    return unique


# ---------------------------------------------------------------------------
# 4. LLM Providers
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Call local Ollama using OpenAI-compatible API."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3",
        base_url: str = "http://localhost:11434/v1",
    ):
        super().__init__(provider="ollama", model=model)
        self.base_url = base_url.rstrip("/")

    def generate_response(self, *, system: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        if not resp.ok:
            err_msg = resp.text
            try:
                err = resp.json()
                if isinstance(err.get("error"), dict):
                    err_msg = err["error"].get("message", err_msg)
                elif isinstance(err.get("error"), str):
                    err_msg = err["error"]
            except Exception:
                pass
            hint = ""
            if "not found" in err_msg.lower():
                hint = " Run `ollama pull <model>` (see LLM_MODEL) to download a model first."
            raise RuntimeError(
                f"Ollama API error ({resp.status_code}): {err_msg}.{hint}"
            ) from None
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class BedrockProvider(LLMProvider):
    """Call Amazon Bedrock Converse API."""

    name = "bedrock"

    def __init__(self, model: str = "qwen3-vl-235b-a22b", region: str = "us-east-1"):
        super().__init__(provider="bedrock", model=model)
        self.region = region
        self._client = None

    def _get_client(self):
        """Lazy-load boto3 client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for BedrockProvider. "
                    "Install it with: pip install boto3"
                ) from None

            self._client = boto3.client(
                service_name="bedrock-runtime", region_name=self.region
            )
        return self._client

    def generate_response(self, *, system: str, user_prompt: str) -> str:
        """Call Bedrock and return model text."""
        client = self._get_client()

        try:
            response = client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                system=[{"text": system}],
                inferenceConfig={
                    "temperature": 0.3,
                    "maxTokens": 4096,
                },
            )

            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content:
                raise RuntimeError("Bedrock returned empty response")

            return content[0].get("text", "")

        except Exception as e:
            error_msg = str(e)
            if "AccessDeniedException" in error_msg:
                raise RuntimeError(
                    "AWS credentials not configured or invalid. "
                    "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables "
                    "or configure ~/.aws/credentials"
                ) from e
            elif (
                "ResourceNotFoundException" in error_msg
                or "ValidationException" in error_msg
            ):
                raise RuntimeError(
                    f"Model '{self.model}' not found in region '{self.region}'. "
                    "Verify model ID and ensure you have access to it in Bedrock."
                ) from e
            else:
                raise RuntimeError(f"Bedrock API error: {error_msg}") from e


def _get_provider_from_env() -> LLMProvider:
    """Build provider from environment settings."""
    provider_name = (os.environ.get("LLM_PROVIDER") or "ollama").lower()
    model = os.environ.get("LLM_MODEL") or "qwen3"
    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"

    if provider_name == "ollama":
        return OllamaProvider(model=model, base_url=base_url)
    if provider_name == "bedrock":
        return BedrockProvider(model=model)
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider_name}. Use 'ollama' or 'bedrock'."
    )


# ---------------------------------------------------------------------------
# 5. Supabase writes
# ---------------------------------------------------------------------------


def _persist_insights(
    sb,
    video_id: str,
    transcript_id: str,
    claims: List[Dict[str, Any]],
    narrative_assignments: List[MatchDecision],
    new_narratives: List[NarrativeCandidate],
) -> int:
    """Insert narratives, claims, and bridge rows."""
    now = datetime.now(timezone.utc).isoformat()

    if new_narratives:
        narr_rows = [
            {
                "narrative_id": str(n.narrative_id),
                "narrative_label": n.narrative_label,
                "narrative_risk_score": n.narrative_risk_score,
                "narrative_category": n.narrative_category,
                "narrative_description": n.narrative_description,
                "narrative_details": n.narrative_details,
                "created_at": now,
            }
            for n in new_narratives
        ]
        sb.table("narratives").insert(narr_rows).execute()

    claim_rows = []
    claim_ids: List[str] = []
    for c in claims:
        cid = str(uuid.uuid4())
        claim_ids.append(cid)
        claim_rows.append(
            {
                "claim_id": cid,
                "video_id": video_id,
                "transcript_id": transcript_id,
                "claim_text": c["text"],
                "llm_confidence": c.get("confidence"),
                "created_at": now,
            }
        )
    if claim_rows:
        sb.table("claims").insert(claim_rows).execute()

    bridge_rows = []
    for cid, decision in zip(claim_ids, narrative_assignments):
        for nid in decision.linked_narrative_ids:
            bridge_rows.append(
                {
                    "claim_id": cid,
                    "narrative_id": str(nid),
                    "created_at": now,
                }
            )
    if bridge_rows:
        sb.table("claim_narratives").insert(bridge_rows).execute()

    return len(claim_rows)


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------


def run_llm_insight_generation_pipeline(
    *,
    provider: LLMProvider | None = None,
) -> dict:
    """Run end-to-end claim extraction and narrative persistence."""
    sb = _get_supabase()
    prov = provider or _get_provider_from_env()

    transcripts = _fetch_transcripts_without_claims(sb)
    if not transcripts:
        log.info("No unprocessed transcripts found.")
        return {"video_ids": [], "total_claims": 0, "total_new_narratives": 0}

    log.info("Processing %d transcript(s)", len(transcripts))

    existing_narr_rows = _fetch_all_narratives(sb)
    embedder = get_embedder_from_env()
    candidates, candidate_embeddings = build_candidate_pool(
        existing_narr_rows, embedder=embedder
    )

    total_claims = 0
    total_new_narratives = 0
    processed_video_ids: List[str] = []

    for t in transcripts:
        video_id = t["video_id"]
        transcript_id = str(t["transcript_id"])
        text = t["cleaned_transcript_txt"]

        try:
            _delete_existing_insights(sb, transcript_id)

            claims = _extract_claims(text, prov)
            if not claims:
                log.info("%s: 0 claims extracted, skipping.", video_id)
                continue

            decisions: List[MatchDecision] = []
            run_new_narratives: List[NarrativeCandidate] = []

            for c in claims:
                decision = match_claim_to_narratives(
                    claim_text=c["text"],
                    narrative_theme=c.get("narrative_theme"),
                    candidates=candidates,
                    candidate_embeddings=candidate_embeddings,
                    embedder=embedder,
                    narrative_category=c.get("narrative_category"),
                    narrative_description=c.get("narrative_description"),
                    narrative_details=c.get("narrative_details"),
                )
                decisions.append(decision)
                log.debug(
                    "Claim match top_similarity=%s",
                    decision.top_similarity,
                )
                if decision.new_narrative:
                    run_new_narratives.append(decision.new_narrative)
                    candidate_embeddings = refresh_pool_with_new(
                        candidates,
                        candidate_embeddings,
                        decision.new_narrative,
                        embedder=embedder,
                    )

            inserted = _persist_insights(
                sb, video_id, transcript_id, claims, decisions, run_new_narratives
            )
            total_claims += inserted
            total_new_narratives += len(run_new_narratives)
            processed_video_ids.append(video_id)
            log.info(
                "%s: %d claims, %d new narratives",
                video_id,
                inserted,
                len(run_new_narratives),
            )
        except Exception:
            log.exception("Failed to process %s", video_id)

    return {
        "video_ids": processed_video_ids,
        "total_claims": total_claims,
        "total_new_narratives": total_new_narratives,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    result = run_llm_insight_generation_pipeline()
    log.info("Pipeline complete: %s", result)
