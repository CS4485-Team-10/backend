import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class Narrative(SQLModel, table=True):
    __tablename__ = "narratives"

    narrative_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    narrative_label: str
    narrative_risk: str
    narrative_description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
