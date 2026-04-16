from datetime import datetime
from typing import Any

from pydantic import BaseModel


class VideoWithChannel(BaseModel):
    video_id: str
    channel_id: str
    title: str
    description: str | None = None
    view_count: int = 0
    published_at: datetime
    thumbnail_url: str | None = None
    stats: dict[str, Any] = {}
    created_at: datetime
    channel_title: str | None = None
    channel_handle: str | None = None
