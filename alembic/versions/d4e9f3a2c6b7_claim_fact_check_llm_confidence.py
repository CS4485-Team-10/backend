"""add fact_check_status and llm_confidence to claims

Revision ID: d4e9f3a2c6b7
Revises: c3d8f2a1b4e5
Create Date: 2026-03-26

"""

from alembic import op
import sqlalchemy as sa

revision = "d4e9f3a2c6b7"
down_revision = "c3d8f2a1b4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "claims",
        sa.Column("fact_check_status", sa.Text(), nullable=True),
    )
    op.add_column(
        "claims",
        sa.Column("llm_confidence", sa.Numeric(5, 4), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("claims", "llm_confidence")
    op.drop_column("claims", "fact_check_status")
