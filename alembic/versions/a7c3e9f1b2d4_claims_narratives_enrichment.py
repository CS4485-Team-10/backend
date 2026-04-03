"""claims risk fields; narratives risk score category details drop narrative_risk

Revision ID: a7c3e9f1b2d4
Revises: f8a1c2d3e4b5
Create Date: 2026-04-02

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision = "a7c3e9f1b2d4"
down_revision = "f8a1c2d3e4b5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("claims", sa.Column("risk_level", sa.Text(), nullable=True))
    op.add_column(
        "claims", sa.Column("fact_check_confidence", sa.Text(), nullable=True)
    )

    op.add_column(
        "narratives",
        sa.Column(
            "narrative_risk_score",
            sa.Numeric(4, 2),
            nullable=False,
            server_default="5.0",
        ),
    )
    op.execute(
        text(
            """
            UPDATE narratives SET narrative_risk_score = CASE LOWER(narrative_risk)
                WHEN 'high' THEN 8.0
                WHEN 'medium' THEN 5.0
                WHEN 'low' THEN 2.0
                ELSE 5.0
            END
            """
        )
    )
    op.create_check_constraint(
        "ck_narratives_narrative_risk_score_range",
        "narratives",
        "narrative_risk_score >= 0 AND narrative_risk_score <= 10",
    )

    op.add_column(
        "narratives",
        sa.Column(
            "narrative_category",
            sa.Text(),
            nullable=False,
            server_default="Uncategorized",
        ),
    )
    op.add_column(
        "narratives",
        sa.Column("narrative_details", sa.Text(), nullable=True),
    )

    op.drop_column("narratives", "narrative_risk")

    op.alter_column(
        "narratives",
        "narrative_risk_score",
        server_default=None,
    )
    op.alter_column(
        "narratives",
        "narrative_category",
        server_default=None,
    )


def downgrade() -> None:
    op.add_column(
        "narratives",
        sa.Column(
            "narrative_risk",
            sa.Text(),
            nullable=False,
            server_default="medium",
        ),
    )
    op.execute(
        text(
            """
            UPDATE narratives SET narrative_risk = CASE
                WHEN narrative_risk_score >= 7.0 THEN 'high'
                WHEN narrative_risk_score >= 4.0 THEN 'medium'
                ELSE 'low'
            END
            """
        )
    )
    op.drop_constraint(
        "ck_narratives_narrative_risk_score_range",
        "narratives",
        type_="check",
    )
    op.drop_column("narratives", "narrative_details")
    op.drop_column("narratives", "narrative_category")
    op.drop_column("narratives", "narrative_risk_score")
    op.alter_column("narratives", "narrative_risk", server_default=None)

    op.drop_column("claims", "fact_check_confidence")
    op.drop_column("claims", "risk_level")
