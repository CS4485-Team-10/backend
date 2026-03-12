from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class Channel(SQLModel, table=True):
    __tablename__ = "channels"

    channel_id: str = Field(primary_key=True)
    title: str
    handle: str
    url: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
