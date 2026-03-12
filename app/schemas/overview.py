from pydantic import BaseModel


class OverviewStatsResponse(BaseModel):
    ok: bool = True
    total_videos_scoped: int
    active_narratives: int
    total_claims: int
    verified_claims: int | None = None
    high_risk_alerts: int | None = None
