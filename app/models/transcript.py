from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel, Text, Column


class Transcript(SQLModel, table=True):
    __tablename__ = "transcripts"

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: str = Field(foreign_key="videos.video_id")
    language: str
    content: str = Field(sa_column=Column(Text))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
