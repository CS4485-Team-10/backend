from datetime import UTC, datetime

from sqlmodel import Column, Field, SQLModel, Text


class Transcript(SQLModel, table=True):
    __tablename__ = "transcripts"

    id: int | None = Field(default=None, primary_key=True)
    video_id: str = Field(foreign_key="videos.video_id")
    language: str
    content: str = Field(sa_column=Column(Text))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
