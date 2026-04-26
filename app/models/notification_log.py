import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class NotificationLog(SQLModel, table=True):
    __tablename__ = "notification_log"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    narrative_id: uuid.UUID = Field(index=True)
    recipient_email: str
    sent_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
