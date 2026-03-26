from typing import Any, Optional

from pydantic import BaseModel


class OverviewStatsResponse(BaseModel):
    ok: bool = True
    total_videos_scoped: int
    active_narratives: int
    total_claims: int
    verified_claims: Optional[int] = None
    high_risk_alerts: Optional[int] = None


class ExecutiveOverviewResponse(BaseModel):
    ok: bool = True
    total_videos_scoped: int
    active_narratives: int
    verified_claims: int
    high_risk_alerts: int
    overview_metrics_meta: dict[str, Any]
    topic_trends_meta: dict[str, Any]
    topic_trends: list[dict[str, Any]]


class TopicCluster(BaseModel):
    label: str
    size: int
    x: float
    y: float


class TopicClustersResponse(BaseModel):
    ok: bool = True
    clusters: list[TopicCluster]
