from datetime import datetime, timezone
from typing import Any, Optional

from sqlmodel import Column, Field, JSON, SQLModel


class Insight(SQLModel, table=True):
    __tablename__ = "insights"

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: str = Field(foreign_key="videos.video_id")
    model: str
    claims: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    narratives: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    labels: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    confidence: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
