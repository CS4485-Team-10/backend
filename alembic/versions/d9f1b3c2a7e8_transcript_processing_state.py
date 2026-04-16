"""add processing_status, last_attempted_at, attempt_count to transcripts

Revision ID: d9f1b3c2a7e8
Revises: c1a4e8f2b9d0
Create Date: 2026-04-16

"""

from alembic import op
import sqlalchemy as sa


revision = "d9f1b3c2a7e8"
down_revision = "c1a4e8f2b9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add columns with temporary server defaults so existing rows backfill safely.
    op.add_column(
        "transcripts",
        sa.Column(
            "processing_status",
            sa.Text(),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column(
        "transcripts",
        sa.Column("last_attempted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "transcripts",
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )

    # Drop the temporary server defaults so the ORM owns defaults going forward,
    # consistent with the rest of this codebase's Python-side defaults.
    op.alter_column("transcripts", "processing_status", server_default=None)
    op.alter_column("transcripts", "attempt_count", server_default=None)


def downgrade() -> None:
    op.drop_column("transcripts", "attempt_count")
    op.drop_column("transcripts", "last_attempted_at")
    op.drop_column("transcripts", "processing_status")
