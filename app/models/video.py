from datetime import datetime, timezone
from typing import Any, Optional

from sqlmodel import Column, Field, JSON, SQLModel


class Video(SQLModel, table=True):
    __tablename__ = "videos"

    video_id: str = Field(primary_key=True)
    channel_id: str = Field(foreign_key="channels.channel_id")
    title: str
    description: Optional[str] = None
    view_count: int = 0
    published_at: datetime
    thumbnail_url: str
    stats: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
