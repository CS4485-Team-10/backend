"""drop comments table and raw_transcript_json from transcripts

Revision ID: f8a1c2d3e4b5
Revises: e5f0a3b7d8c1
Create Date: 2026-04-02

"""

from alembic import op
import sqlalchemy as sa


revision = "f8a1c2d3e4b5"
down_revision = "e5f0a3b7d8c1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("comments")
    op.drop_column("transcripts", "raw_transcript_json")


def downgrade() -> None:
    op.add_column(
        "transcripts",
        sa.Column(
            "raw_transcript_json",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
    )
    op.alter_column("transcripts", "raw_transcript_json", server_default=None)
    op.create_table(
        "comments",
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("comment_threads_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["videos.video_id"],
        ),
        sa.PrimaryKeyConstraint("video_id"),
    )
