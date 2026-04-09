"""drop supporting_excerpt and time columns from claims

Revision ID: c1a4e8f2b9d0
Revises: a7c3e9f1b2d4
Create Date: 2026-04-08

"""

from alembic import op
import sqlalchemy as sa


revision = "c1a4e8f2b9d0"
down_revision = "a7c3e9f1b2d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("claims", "supporting_excerpt")
    op.drop_column("claims", "start_time_seconds")
    op.drop_column("claims", "end_time_seconds")


def downgrade() -> None:
    op.add_column("claims", sa.Column("supporting_excerpt", sa.Text(), nullable=True))
    op.add_column(
        "claims", sa.Column("start_time_seconds", sa.Integer(), nullable=True)
    )
    op.add_column("claims", sa.Column("end_time_seconds", sa.Integer(), nullable=True))
