"""add sentiment_label and sentiment_score to claims

Revision ID: c3d8f2a1b4e5
Revises: b2c7e4f1a9d0
Create Date: 2026-03-26

"""

from alembic import op
import sqlalchemy as sa

revision = "c3d8f2a1b4e5"
down_revision = "b2c7e4f1a9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "claims",
        sa.Column("sentiment_label", sa.Text(), nullable=True),
    )
    op.add_column(
        "claims",
        sa.Column("sentiment_score", sa.Numeric(4, 3), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("claims", "sentiment_score")
    op.drop_column("claims", "sentiment_label")
