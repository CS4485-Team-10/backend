"""add notification_log table

Revision ID: g1b2c3d4e5f6
Revises: f8a1c2d3e4b5
Create Date: 2026-04-16

"""

import sqlalchemy as sa

from alembic import op

revision = "g1b2c3d4e5f6"
down_revision = "c1a4e8f2b9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notification_log",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("narrative_id", sa.Uuid(), nullable=False),
        sa.Column("recipient_email", sa.String(), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_notification_log_narrative_id", "notification_log", ["narrative_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_notification_log_narrative_id", table_name="notification_log")
    op.drop_table("notification_log")
