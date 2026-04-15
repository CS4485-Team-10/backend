from pydantic import BaseModel


class ClaimResponse(BaseModel):
    claim_id: str
    text: str
    source: str
    status: str
    confidence: str
    views: str
    date: str


class ClaimStatsResponse(BaseModel):
    total: int
    verified: int
    disputed: int
    under_review: int


class ClaimListResponse(BaseModel):
    ok: bool = True
    data: list[ClaimResponse]
    stats: ClaimStatsResponse
    count: int
