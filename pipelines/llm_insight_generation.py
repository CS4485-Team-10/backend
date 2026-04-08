"""
LLM Insight Generation Pipeline.

Reads transcripts from Supabase (only those without existing claims),
extracts claims via an LLM, semantically matches or creates narratives,
and persists claims, narratives, and claim_narratives back to Supabase.
Narrative embeddings are only used for semantic matching/dedup against
existing narrative rows, not for LLM extraction. Production defaults to
``NARR_EMBEDDING_BACKEND=remote`` (Google Gemini ``embedContent``): set
``NARR_EMBEDDING_URL`` to the full ``.../models/<id>:embedContent`` URL,
``NARR_EMBEDDING_API_KEY`` (Google AI key, ``x-goog-api-key``), and optionally
``NARR_EMBEDDING_MODEL``, ``NARR_EMBEDDING_TIMEOUT`` (see
``pipelines.narrative_matching``).

Re-runs are safe: prior claims and bridge rows for a transcript are
deleted before new ones are inserted (replace semantics for CRON jobs).
"""

from __future__ import annotations

import json
import logging
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


def _get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# 1. Supabase reads
# ---------------------------------------------------------------------------


def _fetch_transcripts_without_claims(sb) -> List[Dict[str, Any]]:
    """Return transcripts rows that have no corresponding claims yet.

    Uses Supabase PostgREST: fetch all transcripts, then subtract those
    whose transcript_id already appears in claims (set difference in Python
    since PostgREST has limited subquery support).
    """
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

    return [
        t for t in all_transcripts if str(t["transcript_id"]) not in claimed_ids
    ]


def _fetch_all_narratives(sb) -> List[Dict[str, Any]]:
    """Paginated fetch of all rows from the narratives table."""
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
# 2. Replace-semantics: delete old insights for a transcript
# ---------------------------------------------------------------------------


def _delete_existing_insights(sb, transcript_id: str) -> None:
    """Remove claim_narratives and claims for *transcript_id* (FK-safe order)."""
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
    """Split long transcript text into chunks to fit model context limits."""
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


_SYSTEM = "You extract factual claims from the transcript. Return ONLY valid JSON."
_SCHEMA_HINT = """
{
  "claims": [
    {
      "text": "string",
      "confidence": 0.0,
      "narrative_theme": "optional short theme"
    }
  ]
}
""".strip()


def _build_user_prompt(transcript_chunk: str) -> str:
    return f"""Extract the main factual claims from the following transcript.

Expected JSON output format (return ONLY valid JSON, no other text):
{_SCHEMA_HINT}

Transcript:
---
{transcript_chunk}
---

Return a single JSON object with a "claims" array. Each claim must have:
- "text": the factual claim (string)
- "confidence": how confident you are in this claim (0.0 to 1.0)
- "narrative_theme": a short phrase describing the overarching narrative this claim belongs to (optional but preferred)"""


def _validate_json_output(text: str) -> Dict[str, Any]:
    """Parse and validate the LLM JSON output."""
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
    """Chunk transcript, call LLM per chunk, dedupe and return claims."""
    chunks = _chunk_text(transcript_text, max_chars=max_chars)
    all_claims: List[Dict[str, Any]] = []

    for chunk in chunks:
        user_prompt = _build_user_prompt(chunk)
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                raw = provider.generate_response(
                    system=_SYSTEM, user_prompt=user_prompt
                )
                parsed = _validate_json_output(raw)
                all_claims.extend(parsed.get("claims", []))
                break
            except Exception as e:
                last_error = e
                if attempt == retries:
                    raise last_error from last_error

    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for c in all_claims:
        t = c.get("text", "")
        if t and t not in seen:
            seen.add(t)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# 4. LLM Providers
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Calls local Ollama using an OpenAI-compatible endpoint."""

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
                hint = (
                    " Run `ollama pull <model>` (see LLM_MODEL) to download a model first."
                )
            raise RuntimeError(
                f"Ollama API error ({resp.status_code}): {err_msg}.{hint}"
            ) from None
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class BedrockProvider(LLMProvider):
    """Calls Amazon Bedrock using the Converse API. Placeholder for future implementation."""

    name = "bedrock"

    def __init__(self, model: str, region: str = "us-east-1"):
        super().__init__(provider="bedrock", model=model)
        self.region = region

    def generate_response(self, *, system: str, user_prompt: str) -> str:
        raise NotImplementedError("BedrockProvider not yet implemented")


def _get_provider_from_env() -> LLMProvider:
    """Create LLM provider from LLM_PROVIDER and LLM_MODEL env vars."""
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
    """Insert narratives, claims, and claim_narratives to Supabase.

    Returns the number of claims inserted.
    """
    now = datetime.now(timezone.utc).isoformat()

    # 1. Insert new narratives
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

    # 2. Insert claims
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

    # 3. Insert claim_narratives bridge rows
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
    """Main entrypoint for the LLM insight generation pipeline.

    Reads transcripts from Supabase that have no claims yet, runs LLM
    extraction, matches/creates narratives, and persists all rows.
    Safe for repeated / CRON invocations: existing claims for a
    transcript are deleted before re-inserting (replace semantics).

    Run locally:
        conda activate yt-intel-project
        python -m pipelines.llm_insight_generation

    Returns
    -------
    dict with keys: video_ids, total_claims, total_new_narratives
    """
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
