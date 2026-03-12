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
    thumbnail_url: str
    stats: dict[str, Any] = {}
    created_at: datetime
    channel_title: str
    channel_handle: str
