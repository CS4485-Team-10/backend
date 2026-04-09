"""drop channels table and videos.channel_id FK to channels

Revision ID: b2c7e4f1a9d0
Revises: 9a0a89dc9b39
Create Date: 2026-03-25

"""

from alembic import op
import sqlalchemy as sa
import sqlmodel

revision = "b2c7e4f1a9d0"
down_revision = "9a0a89dc9b39"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys("videos"):
        if fk.get("referred_table") == "channels":
            op.drop_constraint(fk["name"], "videos", type_="foreignkey")
            break
    op.drop_table("channels")


def downgrade() -> None:
    op.create_table(
        "channels",
        sa.Column("channel_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("handle", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("url", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("channel_id"),
    )
    op.create_foreign_key(
        "videos_channel_id_fkey",
        "videos",
        "channels",
        ["channel_id"],
        ["channel_id"],
    )
