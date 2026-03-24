"""Shared data models and interfaces for pipeline modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptRecord:
    """Represents one video transcript with paths to cleaned and raw files."""

    video_id: str
    cleaned_txt_path: Path
    raw_json_path: Optional[Path] = None


@dataclass
class GeneratedInsights:
    """LLM-generated outputs plus video metadata for storage."""

    video_id: str
    claims: List[Dict[str, Any]]
    narratives: List[Dict[str, Any]]
    model: str
    provider: str
    source_cleaned_txt: str
    source_raw_json: Optional[str]
    chunk_count: int


class LLMProvider(ABC):
    """Base interface for LLM providers. Implementations must define provider, model, and generate_response."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    @abstractmethod
    def generate_response(self, *, system: str, user_prompt: str) -> str:
        """Generate a response from the LLM. Returns raw string (expected to be JSON)."""
        pass
