import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Column, Field, SQLModel, Text


class Claim(SQLModel, table=True):
    __tablename__ = "claims"

    claim_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    video_id: str = Field(foreign_key="videos.video_id")
    transcript_id: uuid.UUID = Field(foreign_key="transcripts.transcript_id")
    claim_text: str = Field(sa_column=Column(Text, nullable=False))
    supporting_excerpt: Optional[str] = Field(default=None, sa_column=Column(Text))
    start_time_seconds: Optional[int] = None
    end_time_seconds: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
