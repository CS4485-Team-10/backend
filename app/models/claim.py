import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Numeric
from sqlmodel import Column, Field, SQLModel, Text


class Claim(SQLModel, table=True):
    __tablename__ = "claims"

    claim_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    video_id: str = Field(foreign_key="videos.video_id")
    transcript_id: uuid.UUID = Field(foreign_key="transcripts.transcript_id")
    claim_text: str = Field(sa_column=Column(Text, nullable=False))
    sentiment_label: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    sentiment_score: Optional[float] = Field(
        default=None,
        sa_column=Column(Numeric(4, 3), nullable=True),
    )
    fact_check_status: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Optional: "high" | "medium" | "low"
    risk_level: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    fact_check_confidence: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    llm_confidence: Optional[float] = Field(
        default=None,
        sa_column=Column(Numeric(5, 4), nullable=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
