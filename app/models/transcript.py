import uuid
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Column, Field, JSON, SQLModel, Text


class Transcript(SQLModel, table=True):
    __tablename__ = "transcripts"

    transcript_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    video_id: str = Field(foreign_key="videos.video_id", unique=True)
    cleaned_transcript_txt: str = Field(sa_column=Column(Text, nullable=False))
    raw_transcript_json: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
