"""add email_subscribers table

Revision ID: i3b4c5d6e7f8
Revises: h2a3b4c5d6e7
Create Date: 2026-04-23

"""

import sqlalchemy as sa

from alembic import op

revision = "i3b4c5d6e7f8"
down_revision = "h2a3b4c5d6e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "email_subscribers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("subscribed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_email_subscribers_email", "email_subscribers", ["email"])


def downgrade() -> None:
    op.drop_index("ix_email_subscribers_email", table_name="email_subscribers")
    op.drop_table("email_subscribers")
