from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import BigInteger
from sqlmodel import Column, Field, SQLModel


class Video(SQLModel, table=True):
    __tablename__ = "videos"

    video_id: str = Field(primary_key=True)
    channel_id: str = Field(foreign_key="channels.channel_id")
    channel_title: Optional[str] = None
    title: str
    category_id: Optional[str] = None
    published_at: datetime
    view_count: Optional[int] = Field(default=None, sa_column=Column(BigInteger))
    like_count: Optional[int] = Field(default=None, sa_column=Column(BigInteger))
    comment_count: Optional[int] = Field(default=None, sa_column=Column(BigInteger))
    duration_seconds: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
