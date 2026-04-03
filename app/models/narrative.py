import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, Numeric
from sqlmodel import Field, SQLModel, Text


class Narrative(SQLModel, table=True):
    __tablename__ = "narratives"

    narrative_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    narrative_label: str
    # 0.0–10.0 (enforced in DB via CHECK)
    narrative_risk_score: float = Field(
        default=5.0,
        sa_column=Column(Numeric(4, 2), nullable=False),
    )
    narrative_category: str = Field(default="Uncategorized")
    narrative_description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    narrative_details: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
