from datetime import datetime, timezone
from typing import Any

from sqlmodel import Column, Field, JSON, SQLModel


class Comment(SQLModel, table=True):
    __tablename__ = "comments"

    video_id: str = Field(primary_key=True, foreign_key="videos.video_id")
    comment_threads_json: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
