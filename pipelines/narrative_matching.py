"""Semantic narrative matching for claim → narrative assignment.

Compares each claim against existing narratives using sentence embeddings
and cosine similarity, then decides whether to reuse existing narratives
or create new ones. Supports many-to-many linking (one claim can map to
multiple narratives).

Embedding model: all-MiniLM-L6-v2 (fast, 384-dim, good for short texts).
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (overridable via env)
# ---------------------------------------------------------------------------

STRONG_MATCH: float = float(os.environ.get("NARR_STRONG_MATCH", "0.82"))
MULTI_LINK_THRESHOLD: float = float(os.environ.get("NARR_MULTI_LINK", "0.70"))
NEW_NARRATIVE_MIN: float = float(os.environ.get("NARR_NEW_MIN", "0.72"))
MAX_NARRATIVES_PER_CLAIM: int = int(os.environ.get("NARR_MAX_PER_CLAIM", "5"))

EMBEDDING_MODEL_NAME: str = os.environ.get(
    "NARR_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NarrativeCandidate:
    """An existing or newly-created narrative with its text for embedding."""

    narrative_id: uuid.UUID
    narrative_label: str
    narrative_risk: str
    narrative_description: Optional[str] = None
    is_new: bool = False

    @property
    def embed_text(self) -> str:
        parts = [self.narrative_label]
        if self.narrative_description:
            parts.append(self.narrative_description)
        return " — ".join(parts)


@dataclass
class MatchDecision:
    """Result of matching one claim against the narrative pool."""

    linked_narrative_ids: List[uuid.UUID] = field(default_factory=list)
    new_narrative: Optional[NarrativeCandidate] = None


# ---------------------------------------------------------------------------
# Embedder (lazy singleton)
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings (N x D) for a list of strings."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarities between a single vector *a* (1-D) and matrix *b* (N x D).

    Both inputs are assumed L2-normalized, so dot product == cosine similarity.
    """
    return b @ a


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------


def match_claim_to_narratives(
    claim_text: str,
    narrative_theme: Optional[str],
    candidates: List[NarrativeCandidate],
    candidate_embeddings: np.ndarray,
) -> MatchDecision:
    """Decide which existing narratives a claim links to, or create a new one.

    Parameters
    ----------
    claim_text:
        The raw claim text extracted by the LLM.
    narrative_theme:
        Optional short theme string the LLM may have provided for this claim.
    candidates:
        All known narratives (existing DB rows + any created during this run).
    candidate_embeddings:
        Pre-computed embeddings for *candidates*, same order / length.

    Returns
    -------
    MatchDecision with linked IDs and optionally a new NarrativeCandidate.
    """
    if not candidates:
        new_narr = _build_new_narrative(claim_text, narrative_theme)
        return MatchDecision(linked_narrative_ids=[new_narr.narrative_id], new_narrative=new_narr)

    query = claim_text if not narrative_theme else f"{claim_text} — {narrative_theme}"
    query_vec = embed([query])[0]
    sims = cosine_sim(query_vec, candidate_embeddings)

    max_sim = float(np.max(sims))
    linked_ids: List[uuid.UUID] = []

    if max_sim >= MULTI_LINK_THRESHOLD:
        top_indices = np.where(sims >= MULTI_LINK_THRESHOLD)[0]
        ranked = sorted(top_indices, key=lambda i: sims[i], reverse=True)
        for idx in ranked[:MAX_NARRATIVES_PER_CLAIM]:
            linked_ids.append(candidates[idx].narrative_id)

    if max_sim < NEW_NARRATIVE_MIN:
        new_narr = _build_new_narrative(claim_text, narrative_theme)
        linked_ids.append(new_narr.narrative_id)
        return MatchDecision(linked_narrative_ids=linked_ids, new_narrative=new_narr)

    return MatchDecision(linked_narrative_ids=linked_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_new_narrative(
    claim_text: str, theme: Optional[str]
) -> NarrativeCandidate:
    label = theme if theme else claim_text[:120]
    return NarrativeCandidate(
        narrative_id=uuid.uuid4(),
        narrative_label=label,
        narrative_risk="medium",
        narrative_description=claim_text[:500] if not theme else claim_text[:500],
        is_new=True,
    )


def build_candidate_pool(
    existing_rows: List[Dict[str, Any]],
) -> tuple[List[NarrativeCandidate], np.ndarray]:
    """Convert Supabase narrative rows into candidates + their embeddings.

    Parameters
    ----------
    existing_rows:
        Dicts with at least ``narrative_id``, ``narrative_label``,
        ``narrative_risk``, and optionally ``narrative_description``.

    Returns
    -------
    (candidates, embeddings) — embeddings are L2-normalized (N x D).
    Returns empty arrays when there are no existing rows.
    """
    candidates = [
        NarrativeCandidate(
            narrative_id=uuid.UUID(r["narrative_id"]) if isinstance(r["narrative_id"], str) else r["narrative_id"],
            narrative_label=r["narrative_label"],
            narrative_risk=r.get("narrative_risk", "medium"),
            narrative_description=r.get("narrative_description"),
        )
        for r in existing_rows
    ]
    if not candidates:
        return candidates, np.empty((0, 0))

    texts = [c.embed_text for c in candidates]
    embeddings = embed(texts)
    return candidates, embeddings


def refresh_pool_with_new(
    candidates: List[NarrativeCandidate],
    embeddings: np.ndarray,
    new_narrative: NarrativeCandidate,
) -> np.ndarray:
    """Append a newly-created narrative to the in-memory pool.

    Mutates *candidates* in-place and returns the updated embeddings matrix.
    """
    candidates.append(new_narrative)
    new_emb = embed([new_narrative.embed_text])
    if embeddings.size == 0:
        return new_emb
    return np.vstack([embeddings, new_emb])
