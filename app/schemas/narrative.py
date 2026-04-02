from typing import Optional

from pydantic import BaseModel


class NarrativeResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    detail: Optional[str]
    category: str
    risk_level: str
    risk_score: float
    videos_analyzed: int
    total_views: str
    time_window: str
    primary_link: Optional[str]


class NarrativeListResponse(BaseModel):
    ok: bool = True
    data: list[NarrativeResponse]
    count: int
