import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Column, Field, SQLModel, Text


class Transcript(SQLModel, table=True):
    __tablename__ = "transcripts"

    transcript_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    video_id: str = Field(foreign_key="videos.video_id", unique=True)
    cleaned_transcript_txt: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    processing_status: str = Field(
        default="pending",
        description=(
            "pending (fresh ingest), pending_completion (retry after timeout/error), "
            "in_progress, done"
        ),
        sa_column=Column(Text, nullable=False),
    )
    last_attempted_at: Optional[datetime] = Field(default=None, nullable=True)
    attempt_count: int = Field(default=0, nullable=False)
