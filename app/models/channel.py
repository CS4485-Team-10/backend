from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Channel(SQLModel, table=True):
    __tablename__ = "channels"

    channel_id: str = Field(primary_key=True)
    title: str
    handle: str
    url: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
