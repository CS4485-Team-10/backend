import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class EmailSubscriber(SQLModel, table=True):
    __tablename__ = "email_subscribers"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    subscribed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notify_high_risk: bool = Field(default=True)
    notify_medium_risk: bool = Field(default=False)
    is_active: bool = Field(default=True)
