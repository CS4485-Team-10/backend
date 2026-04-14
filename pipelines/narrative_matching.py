"""Match claims to existing narratives or create new ones."""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (overridable via env)
# ---------------------------------------------------------------------------

STRONG_MATCH: float = float(os.environ.get("NARR_STRONG_MATCH", "0.82"))
MULTI_LINK_THRESHOLD: float = float(os.environ.get("NARR_MULTI_LINK", "0.70"))
NEW_NARRATIVE_MIN: float = float(os.environ.get("NARR_NEW_MIN", "0.72"))
MAX_NARRATIVES_PER_CLAIM: int = int(os.environ.get("NARR_MAX_PER_CLAIM", "5"))

_DEFAULT_EMBEDDING_TIMEOUT = 60.0


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def _validate_embeddings(texts: List[str], arr: np.ndarray) -> None:
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {arr.shape}")
    if arr.shape[0] != len(texts):
        raise ValueError(f"expected {len(texts)} embedding rows, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError("embeddings contain non-finite values")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NarrativeCandidate:
    """Narrative candidate used during matching."""

    narrative_id: uuid.UUID
    narrative_label: str
    narrative_risk_score: float = 5.0
    narrative_category: str = "Uncategorized"
    narrative_description: Optional[str] = None
    narrative_details: Optional[str] = None
    is_new: bool = False

    @property
    def embed_text(self) -> str:
        """Build embedding text from narrative fields."""
        parts = [self.narrative_label, self.narrative_category]
        if self.narrative_description:
            parts.append(self.narrative_description)
        if self.narrative_details:
            parts.append(self.narrative_details)
        return " — ".join(parts)


@dataclass
class MatchDecision:
    """Result for one claim match."""

    linked_narrative_ids: List[uuid.UUID] = field(default_factory=list)
    new_narrative: Optional[NarrativeCandidate] = None
    top_similarity: Optional[float] = None


# ---------------------------------------------------------------------------
# Embedder backends
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Text encoder interface."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings."""


def _gemini_error_detail(resp: Any) -> str:
    """Extract a short Gemini error string."""
    try:
        data = resp.json()
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:500]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return (getattr(resp, "text", None) or "")[:500]


class RemoteEmbedder(BaseEmbedder):
    """Gemini embedContent client."""

    def __init__(
        self,
        url: str,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = _DEFAULT_EMBEDDING_TIMEOUT,
    ) -> None:
        import requests as requests_lib

        self._requests = requests_lib
        self._url = url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds

    def _embed_one(self, text: str) -> np.ndarray:
        """Call Gemini once and return one raw vector."""
        body: Dict[str, Any] = {
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY",
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        try:
            resp = self._requests.post(
                self._url,
                json=body,
                headers=headers,
                timeout=self._timeout,
            )
        except self._requests.RequestException as e:
            raise RuntimeError(
                f"Gemini embedContent request failed ({self._url!r}): {e}"
            ) from e
        if not resp.ok:
            detail = _gemini_error_detail(resp)
            raise RuntimeError(
                f"Gemini embedContent HTTP {resp.status_code} ({self._url!r}): {detail}"
            )
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Gemini embedContent response is not JSON ({self._url!r}): {e}"
            ) from e
        if isinstance(data.get("error"), dict):
            msg = data["error"].get("message", str(data["error"]))
            raise RuntimeError(f"Gemini embedContent API error ({self._url!r}): {msg}")
        emb = data.get("embedding")
        if not isinstance(emb, dict):
            raise RuntimeError(
                f"Gemini response missing 'embedding' object ({self._url!r})"
            )
        values = emb.get("values")
        if values is None:
            raise RuntimeError(
                f"Gemini response missing 'embedding.values' ({self._url!r})"
            )
        vec = np.asarray(values, dtype=np.float64)
        if vec.ndim != 1:
            raise RuntimeError(
                f"Gemini embedding must be 1-D, got shape {vec.shape} ({self._url!r})"
            )
        if not np.isfinite(vec).all():
            raise RuntimeError(
                f"Gemini embedding has non-finite values ({self._url!r})"
            )
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        log.debug("Gemini embedContent: %d text(s)", len(texts))
        rows: List[np.ndarray] = []
        dim: Optional[int] = None
        for t in texts:
            vec = self._embed_one(t)
            if dim is None:
                dim = int(vec.shape[0])
            elif int(vec.shape[0]) != dim:
                raise RuntimeError(
                    f"Inconsistent Gemini embedding dimensions: expected {dim}, "
                    f"got {int(vec.shape[0])} ({self._url!r})"
                )
            rows.append(vec)
        arr = np.stack(rows, axis=0)
        try:
            _validate_embeddings(texts, arr)
        except ValueError as e:
            raise RuntimeError(
                f"Invalid Gemini embedding matrix ({self._url!r}): {e}"
            ) from e
        return _l2_normalize_rows(arr)


def get_embedder_from_env() -> BaseEmbedder:
    backend = (
        (os.environ.get("NARR_EMBEDDING_BACKEND") or "remote").lower().replace("-", "_")
    )
    if backend != "remote":
        raise ValueError(
            f"Unsupported NARR_EMBEDDING_BACKEND: {backend!r}. "
            "Only 'remote' (Google Gemini embedContent) is supported."
        )
    timeout = float(
        os.environ.get("NARR_EMBEDDING_TIMEOUT", str(_DEFAULT_EMBEDDING_TIMEOUT))
    )
    api_key = os.environ.get("NARR_EMBEDDING_API_KEY") or None
    url = os.environ.get("NARR_EMBEDDING_URL", "").strip()
    if not url:
        raise ValueError(
            "NARR_EMBEDDING_URL must be set (Gemini embedContent endpoint)."
        )
    remote_model = os.environ.get("NARR_EMBEDDING_MODEL")
    log_model = remote_model if remote_model else "(unset)"
    embedder: BaseEmbedder = RemoteEmbedder(
        url,
        model=remote_model or None,
        api_key=api_key,
        timeout_seconds=timeout,
    )

    log.info(
        "Narrative embedding backend: %s (model=%s)",
        backend,
        log_model,
    )
    log.debug("NARR_EMBEDDING_URL=%s", url)

    return embedder


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity of one vector vs many."""
    return b @ a


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------


def match_claim_to_narratives(
    claim_text: str,
    narrative_theme: Optional[str],
    candidates: List[NarrativeCandidate],
    candidate_embeddings: np.ndarray,
    *,
    embedder: BaseEmbedder,
    narrative_category: Optional[str] = None,
    narrative_description: Optional[str] = None,
    narrative_details: Optional[str] = None,
) -> MatchDecision:
    """Link a claim to existing narratives or create one."""
    if not candidates:
        new_narr = _build_new_narrative(
            claim_text,
            narrative_theme=narrative_theme,
            narrative_category=narrative_category,
            narrative_description=narrative_description,
            narrative_details=narrative_details,
        )
        return MatchDecision(
            linked_narrative_ids=[new_narr.narrative_id], new_narrative=new_narr
        )

    query = claim_text if not narrative_theme else f"{claim_text} — {narrative_theme}"
    query_vec = embedder.encode([query])[0]
    sims = cosine_sim(query_vec, candidate_embeddings)

    max_sim = float(np.max(sims))
    linked_ids: List[uuid.UUID] = []

    if max_sim >= MULTI_LINK_THRESHOLD:
        top_indices = np.where(sims >= MULTI_LINK_THRESHOLD)[0]
        ranked = sorted(top_indices, key=lambda i: sims[i], reverse=True)
        for idx in ranked[:MAX_NARRATIVES_PER_CLAIM]:
            linked_ids.append(candidates[idx].narrative_id)

    if max_sim < NEW_NARRATIVE_MIN:
        new_narr = _build_new_narrative(
            claim_text,
            narrative_theme=narrative_theme,
            narrative_category=narrative_category,
            narrative_description=narrative_description,
            narrative_details=narrative_details,
        )
        linked_ids.append(new_narr.narrative_id)
        return MatchDecision(
            linked_narrative_ids=linked_ids,
            new_narrative=new_narr,
            top_similarity=max_sim,
        )

    return MatchDecision(linked_narrative_ids=linked_ids, top_similarity=max_sim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MAX_FALLBACK_LABEL_CHARS = 120


def _build_new_narrative(
    claim_text: str,
    *,
    narrative_theme: Optional[str] = None,
    narrative_category: Optional[str] = None,
    narrative_description: Optional[str] = None,
    narrative_details: Optional[str] = None,
) -> NarrativeCandidate:
    """Build a new narrative candidate."""
    label = (
        narrative_theme if narrative_theme else claim_text[:_MAX_FALLBACK_LABEL_CHARS]
    )
    category = (narrative_category or "").strip() or "Uncategorized"
    return NarrativeCandidate(
        narrative_id=uuid.uuid4(),
        narrative_label=label,
        narrative_risk_score=5.0,
        narrative_category=category,
        narrative_description=narrative_description,
        narrative_details=narrative_details,
        is_new=True,
    )


def build_candidate_pool(
    existing_rows: List[Dict[str, Any]],
    *,
    embedder: BaseEmbedder,
) -> tuple[List[NarrativeCandidate], np.ndarray]:
    """Convert DB narrative rows to candidates and embeddings."""

    def _risk_score_from_row(r: Dict[str, Any]) -> float:
        v = r.get("narrative_risk_score")
        if v is not None:
            return float(v)
        s = str(r.get("narrative_risk", "medium")).lower()
        return {"high": 8.0, "medium": 5.0, "low": 2.0}.get(s, 5.0)

    candidates = [
        NarrativeCandidate(
            narrative_id=uuid.UUID(r["narrative_id"])
            if isinstance(r["narrative_id"], str)
            else r["narrative_id"],
            narrative_label=r["narrative_label"],
            narrative_risk_score=_risk_score_from_row(r),
            narrative_category=r.get("narrative_category") or "Uncategorized",
            narrative_description=r.get("narrative_description"),
            narrative_details=r.get("narrative_details"),
        )
        for r in existing_rows
    ]
    if not candidates:
        return candidates, np.empty((0, 0))

    texts = [c.embed_text for c in candidates]
    embeddings = embedder.encode(texts)
    return candidates, embeddings


def refresh_pool_with_new(
    candidates: List[NarrativeCandidate],
    embeddings: np.ndarray,
    new_narrative: NarrativeCandidate,
    *,
    embedder: BaseEmbedder,
) -> np.ndarray:
    """Append a new narrative and return updated embeddings."""
    candidates.append(new_narrative)
    new_emb = embedder.encode([new_narrative.embed_text])
    if embeddings.size == 0:
        return new_emb
    return np.vstack([embeddings, new_emb])
