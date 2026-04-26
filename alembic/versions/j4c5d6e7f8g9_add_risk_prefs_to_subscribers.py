"""add risk level preferences to email_subscribers

Revision ID: j4c5d6e7f8g9
Revises: i3b4c5d6e7f8
Create Date: 2026-04-23

"""

import sqlalchemy as sa

from alembic import op

revision = "j4c5d6e7f8g9"
down_revision = "i3b4c5d6e7f8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "email_subscribers",
        sa.Column(
            "notify_high_risk", sa.Boolean(), nullable=False, server_default="true"
        ),
    )
    op.add_column(
        "email_subscribers",
        sa.Column(
            "notify_medium_risk", sa.Boolean(), nullable=False, server_default="false"
        ),
    )


def downgrade() -> None:
    op.drop_column("email_subscribers", "notify_medium_risk")
    op.drop_column("email_subscribers", "notify_high_risk")
