"""add narrative_risk column to narratives

Revision ID: e5f0a3b7d8c1
Revises: d4e9f3a2c6b7
Create Date: 2026-03-26

"""

from alembic import op
import sqlalchemy as sa

revision = "e5f0a3b7d8c1"
down_revision = "d4e9f3a2c6b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "narratives",
        sa.Column(
            "narrative_risk",
            sa.Text(),
            nullable=False,
            server_default="medium",
        ),
    )


def downgrade() -> None:
    op.drop_column("narratives", "narrative_risk")
