import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class ClaimNarrative(SQLModel, table=True):
    __tablename__ = "claim_narratives"

    claim_id: uuid.UUID = Field(primary_key=True, foreign_key="claims.claim_id")
    narrative_id: uuid.UUID = Field(primary_key=True, foreign_key="narratives.narrative_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
