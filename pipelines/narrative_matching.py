"""Match claims to existing narratives or create new ones.

Embeddings come from **AWS Bedrock Titan Text Embeddings V2** via
``bedrock-runtime`` ``invoke_model``. Embeddings are produced and used
in-memory only; nothing is persisted.
"""

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

MULTI_LINK_THRESHOLD: float = float(os.environ.get("NARR_MULTI_LINK", "0.68"))
NEW_NARRATIVE_MIN: float = float(os.environ.get("NARR_NEW_MIN", "0.68"))
MAX_NARRATIVES_PER_CLAIM: int = int(os.environ.get("NARR_MAX_PER_CLAIM", "3"))

_DEFAULT_EMBEDDING_TIMEOUT = 60.0
_DEFAULT_BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
_DEFAULT_BEDROCK_EMBEDDING_DIMENSIONS = 512
_DEFAULT_BEDROCK_EMBEDDING_NORMALIZE = True


def _parse_bool_env(raw: Optional[str], default: bool) -> bool:
    """Parse a permissive truthy/falsey env value."""
    if raw is None:
        return default
    s = raw.strip().lower()
    if not s:
        return default
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean env value: {raw!r}")


def _float_env(name: str, default: float) -> float:
    """Read a float from env, treating missing or blank as ``default``."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as e:
        raise ValueError(f"{name} must be a number.") from e


def _int_env(name: str, default: int) -> int:
    """Read an int from env, treating missing or blank as ``default``."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip(), 10)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer.") from e


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


class BedrockTitanEmbedder(BaseEmbedder):
    """AWS Bedrock Titan Text Embeddings V2 client (``invoke_model``)."""

    def __init__(
        self,
        *,
        model: str,
        region: str,
        dimensions: int,
        normalize: bool,
        timeout_seconds: float = _DEFAULT_EMBEDDING_TIMEOUT,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise ImportError(
                "boto3 is required for BedrockTitanEmbedder. "
                "Install it with: pip install boto3"
            ) from e

        self._model = model
        self._region = region
        self._dimensions = dimensions
        self._normalize = normalize
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                connect_timeout=timeout_seconds,
                read_timeout=timeout_seconds,
            ),
        )

    def _embed_one(self, text: str) -> np.ndarray:
        """Call Titan once and return one raw vector."""
        s = (text or "").strip()
        if not s:
            raise RuntimeError(
                "Cannot embed empty text with Bedrock Titan Text Embeddings V2."
            )

        body = {
            "inputText": s,
            "dimensions": self._dimensions,
            "normalize": self._normalize,
        }

        try:
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as e:
            raise ImportError("botocore is required for BedrockTitanEmbedder.") from e

        try:
            response = self._client.invoke_model(
                modelId=self._model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except ClientError as e:
            error_msg = str(e)
            if "AccessDeniedException" in error_msg:
                raise RuntimeError(
                    f"Bedrock Titan invoke_model access denied (model={self._model!r}, "
                    f"region={self._region!r}): ensure IAM allows "
                    "bedrock:InvokeModel for this model in this region. "
                    "Locally, verify AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
                    "(and optional AWS_SESSION_TOKEN) are loaded."
                ) from e
            if (
                "ResourceNotFoundException" in error_msg
                or "ValidationException" in error_msg
            ):
                raise RuntimeError(
                    f"Bedrock Titan model {self._model!r} not found or invalid "
                    f"in region {self._region!r}: verify the model id and that "
                    "the model is enabled in this region."
                ) from e
            raise RuntimeError(
                f"Bedrock Titan invoke_model failed (model={self._model!r}, "
                f"region={self._region!r}): {error_msg}"
            ) from e
        except BotoCoreError as e:
            raise RuntimeError(
                f"Bedrock Titan invoke_model transport failure (model={self._model!r}, "
                f"region={self._region!r}): {e}"
            ) from e

        try:
            raw = response["body"].read()
            data = json.loads(raw)
        except (KeyError, AttributeError, TypeError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Bedrock Titan response is not valid JSON (model={self._model!r}): {e}"
            ) from e

        embedding = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(
                f"Bedrock Titan response missing non-empty 'embedding' list "
                f"(model={self._model!r})"
            )
        try:
            vec = np.asarray(embedding, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"Bedrock Titan 'embedding' is not numeric (model={self._model!r}): {e}"
            ) from e
        if vec.ndim != 1:
            raise RuntimeError(
                f"Bedrock Titan embedding must be 1-D, got shape {vec.shape} "
                f"(model={self._model!r})"
            )
        if not np.isfinite(vec).all():
            raise RuntimeError(
                f"Bedrock Titan embedding has non-finite values (model={self._model!r})"
            )
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        log.debug("Bedrock Titan invoke_model: %d text(s)", len(texts))
        rows: List[np.ndarray] = []
        dim: Optional[int] = None
        for t in texts:
            vec = self._embed_one(t)
            if dim is None:
                dim = int(vec.shape[0])
            elif int(vec.shape[0]) != dim:
                raise RuntimeError(
                    f"Inconsistent Bedrock Titan embedding dimensions: expected {dim}, "
                    f"got {int(vec.shape[0])} (model={self._model!r})"
                )
            rows.append(vec)
        arr = np.stack(rows, axis=0)
        try:
            _validate_embeddings(texts, arr)
        except ValueError as e:
            raise RuntimeError(
                f"Invalid Bedrock Titan embedding matrix (model={self._model!r}): {e}"
            ) from e
        return _l2_normalize_rows(arr)


def get_embedder_from_env() -> BaseEmbedder:
    """Build the configured embedder from environment variables.

    Only backend is **AWS Bedrock Titan Text Embeddings V2** via
    ``invoke_model``. ``NARR_EMBEDDING_BACKEND`` is optional and, if set, must
    equal ``bedrock``.
    """
    backend = (
        (os.environ.get("NARR_EMBEDDING_BACKEND") or "bedrock")
        .lower()
        .replace("-", "_")
    )
    if backend != "bedrock":
        raise ValueError(
            f"Unsupported NARR_EMBEDDING_BACKEND: {backend!r}. "
            "Only 'bedrock' (AWS Bedrock Titan Text Embeddings V2 via "
            "invoke_model) is supported."
        )

    timeout = _float_env("NARR_EMBEDDING_TIMEOUT", _DEFAULT_EMBEDDING_TIMEOUT)

    region = (
        os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    ).strip()
    if not region:
        raise ValueError(
            "AWS_REGION (or AWS_DEFAULT_REGION) must be set for the Bedrock "
            "Titan embedding backend."
        )
    model_raw = os.environ.get("NARR_EMBEDDING_MODEL")
    model = (
        model_raw if model_raw is not None else _DEFAULT_BEDROCK_EMBEDDING_MODEL
    ).strip()
    if not model:
        raise ValueError(
            "NARR_EMBEDDING_MODEL is set but empty; expected a Bedrock Titan "
            "model id (e.g. 'amazon.titan-embed-text-v2:0')."
        )
    dimensions = _int_env(
        "NARR_EMBEDDING_DIMENSIONS", _DEFAULT_BEDROCK_EMBEDDING_DIMENSIONS
    )
    if dimensions <= 0:
        raise ValueError("NARR_EMBEDDING_DIMENSIONS must be a positive integer.")
    normalize = _parse_bool_env(
        os.environ.get("NARR_EMBEDDING_NORMALIZE"),
        _DEFAULT_BEDROCK_EMBEDDING_NORMALIZE,
    )

    embedder: BaseEmbedder = BedrockTitanEmbedder(
        model=model,
        region=region,
        dimensions=dimensions,
        normalize=normalize,
        timeout_seconds=timeout,
    )

    log.info(
        "Narrative embedding backend: bedrock (model=%s, region=%s, "
        "dimensions=%d, normalize=%s)",
        model,
        region,
        dimensions,
        normalize,
    )
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
