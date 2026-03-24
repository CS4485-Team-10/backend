"""
LLM Insight Generation Pipeline.

Takes transcripts from the ingestion pipeline and passes them into an LLM
to extract claims, narratives, and trends.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import requests

from pipelines.shared import GeneratedInsights, LLMProvider, TranscriptRecord

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

DATA_ROOT = (
    Path.cwd().parent / "data"
    if Path.cwd().name == "pipelines"
    else Path.cwd() / "data"
)


def _resolve_cleaned_dir() -> Path:
    """Resolve path to data/transcripts/cleaned relative to backend root."""
    return DATA_ROOT / "transcripts" / "cleaned"


def _resolve_raw_json_dir() -> Path:
    """Resolve path to data/transcripts/raw relative to backend root."""
    return DATA_ROOT / "transcripts" / "raw"


# ---------------------------------------------------------------------------
# 1. Load & Build Transcripts (private helpers)
# ---------------------------------------------------------------------------


def _load_cleaned_transcript(fp: Union[str, Path]) -> str:
    """Load a single cleaned transcript file and return its text content."""
    file_path = Path(fp)
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    if file_path.suffix != ".txt":
        raise ValueError(f"File is not formatted as a .txt file: {file_path}")
    return file_path.read_text(encoding="utf-8")


def _transcript_record_from_path(fp: Union[str, Path]) -> TranscriptRecord:
    """Build a TranscriptRecord from a single cleaned transcript file path."""
    path = Path(fp).resolve()
    video_id = path.stem
    return TranscriptRecord(
        video_id=video_id,
        cleaned_txt_path=path,
        raw_json_path=None,
    )


# ---------------------------------------------------------------------------
# 2. Prepare for Analysis (private helpers)
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


def _validate_json_output(text: str) -> Dict[str, Any]:
    """Parse and sanity-check the model output as JSON. Requires 'claims' key."""
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


# ---------------------------------------------------------------------------
# 3. Output structure
# ---------------------------------------------------------------------------

_SYSTEM = "You extract factual claims from the transcript. Return ONLY valid JSON."
_SCHEMA_HINT = """
{
  "claims": [{"text": "string", "confidence": 0.0}]
}
""".strip()


def _build_user_prompt(transcript_chunk: str) -> str:
    """Build the user prompt for the LLM, including the expected JSON shape."""
    return f"""Extract the main factual claims from the following transcript.

Expected JSON output format (return ONLY valid JSON, no other text):
{_SCHEMA_HINT}

Transcript:
---
{transcript_chunk}
---

Return your response as a single JSON object with a "claims" array. Each claim should have "text" (the claim) and "confidence" (0.0 to 1.0)."""


def _extract_insights_for_record(
    record: TranscriptRecord,
    provider: LLMProvider,
    *,
    max_chars: int = 12000,
    retries: int = 2,
) -> GeneratedInsights:
    """Load transcript, chunk it, run provider on each chunk, merge and return GeneratedInsights."""
    text = _load_cleaned_transcript(record.cleaned_txt_path)
    chunks = _chunk_text(text, max_chars=max_chars)
    all_claims: List[Dict[str, Any]] = []

    for chunk in chunks:
        user_prompt = _build_user_prompt(chunk)
        last_error = None
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
    unique_claims: List[Dict[str, Any]] = []
    for c in all_claims:
        t = c.get("text", "")
        if t and t not in seen:
            seen.add(t)
            unique_claims.append(c)

    return GeneratedInsights(
        video_id=record.video_id,
        claims=unique_claims,
        narratives=[],
        model=provider.model,
        provider=provider.provider,
        source_cleaned_txt=str(record.cleaned_txt_path),
        source_raw_json=str(record.raw_json_path) if record.raw_json_path else None,
        chunk_count=len(chunks),
    )


# ---------------------------------------------------------------------------
# 4. LLM Providers (public - used for provider selection)
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Calls local Ollama using an OpenAI-compatible endpoint."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3",
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
                hint = " Run `ollama pull llama3` (or another model) to download a model first."
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


# ---------------------------------------------------------------------------
# 5. Private batch helpers
# ---------------------------------------------------------------------------


def _build_transcript_records(
    cleaned_dir: Union[str, Path],
    raw_json_dir: Union[str, Path, None] = None,
) -> List[TranscriptRecord]:
    """Scan cleaned_dir (and optionally raw_json_dir) and build TranscriptRecord for each transcript."""
    cleaned_path = Path(cleaned_dir)
    raw_path = Path(raw_json_dir) if raw_json_dir else None
    records: List[TranscriptRecord] = []
    for fp in sorted(cleaned_path.glob("*.txt")):
        video_id = fp.stem
        raw_json = (
            (raw_path / f"{video_id}.json")
            if raw_path and raw_path.exists()
            else None
        )
        records.append(
            TranscriptRecord(
                video_id=video_id,
                cleaned_txt_path=fp.resolve(),
                raw_json_path=(
                    raw_json.resolve() if raw_json and raw_json.exists() else None
                ),
            )
        )
    return records


def _get_provider_from_env() -> LLMProvider:
    """Create LLM provider from LLM_PROVIDER and LLM_MODEL env vars."""
    provider_name = os.environ.get("LLM_PROVIDER", "ollama").lower()
    model = os.environ.get("LLM_MODEL", "llama3")

    if provider_name == "ollama":
        return OllamaProvider(model=model)
    if provider_name == "bedrock":
        return BedrockProvider(model=model)
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider_name}. Use 'ollama' or 'bedrock'."
    )


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------


def run_llm_insight_generation_pipeline(
    *,
    cleaned_dir: Union[str, Path, None] = None,
    raw_json_dir: Union[str, Path, None] = None,
    provider: LLMProvider | None = None,
    verbose: bool = True,
) -> dict:
    """
    Main entrypoint for the LLM insight generation pipeline.

    Loads transcripts from storage, runs extraction per transcript via the
    configured LLM provider, and returns a results summary. Call this from a
    backend route or run locally via `python -m pipelines.llm_insight_generation`.

    Uses LLM_PROVIDER (ollama|bedrock) and LLM_MODEL env vars when provider
    is not passed.

    Returns:
        dict with keys: insights, video_ids, total_claims
    """
    cleaned = cleaned_dir or _resolve_cleaned_dir()
    raw = raw_json_dir or _resolve_raw_json_dir()

    if not Path(cleaned).exists():
        raise FileNotFoundError(
            f"Cleaned transcripts directory not found: {cleaned} "
            "(run ingestion pipeline first)"
        )

    records = _build_transcript_records(
        cleaned, raw if Path(raw).exists() else None
    )
    if not records:
        return {"insights": [], "video_ids": [], "total_claims": 0}

    prov = provider or _get_provider_from_env()
    insights: List[GeneratedInsights] = []

    for record in records:
        try:
            result = _extract_insights_for_record(record, prov)
            insights.append(result)
            if verbose:
                print(
                    f"  {record.video_id}: {len(result.claims)} claims, "
                    f"{result.chunk_count} chunks"
                )
        except Exception as e:
            if verbose:
                print(f"  {record.video_id}: ERROR - {e}")

    total_claims = sum(len(i.claims) for i in insights)
    return {
        "insights": insights,
        "video_ids": [i.video_id for i in insights],
        "total_claims": total_claims,
    }


if __name__ == "__main__":
    run_llm_insight_generation_pipeline()
