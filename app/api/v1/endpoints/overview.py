import math
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlmodel import Session, col, func, select

from app.core.database import get_session
from app.models.claim import Claim
from app.models.claim_narrative import ClaimNarrative
from app.models.narrative import Narrative
from app.models.video import Video
from app.schemas.overview import (
    ExecutiveOverviewResponse,
    OverviewStatsResponse,
    TopicClustersResponse,
)

router = APIRouter()


@router.get("/overview/stats", response_model=OverviewStatsResponse)
def overview_stats(session: Session = Depends(get_session)):
    total_videos = session.exec(select(func.count()).select_from(Video)).one()
    total_narratives = session.exec(select(func.count()).select_from(Narrative)).one()
    total_claims = session.exec(select(func.count()).select_from(Claim)).one()

    return OverviewStatsResponse(
        total_videos_scoped=total_videos,
        active_narratives=total_narratives,
        total_claims=total_claims,
        verified_claims=None,
        high_risk_alerts=None,
    )


@router.get("/overview/executive", response_model=ExecutiveOverviewResponse)
def executive_overview(session: Session = Depends(get_session)):
    now = datetime.utcnow()
    thirty_days_ago = now - timedelta(days=30)
    sixty_days_ago = now - timedelta(days=60)
    ninety_days_ago = now - timedelta(days=90)

    # ── real data ──────────────────────────────────────────────
    total_videos = session.exec(select(func.count()).select_from(Video)).one()

    videos_current = session.exec(
        select(func.count())
        .select_from(Video)
        .where(Video.created_at >= thirty_days_ago)
    ).one()
    videos_previous = session.exec(
        select(func.count())
        .select_from(Video)
        .where(Video.created_at >= sixty_days_ago, Video.created_at < thirty_days_ago)
    ).one()
    delta_pct = round(
        (videos_current - videos_previous) / max(videos_previous, 1) * 100
    )

    total_narratives = session.exec(select(func.count()).select_from(Narrative)).one()
    new_narratives = session.exec(
        select(func.count())
        .select_from(Narrative)
        .where(Narrative.created_at >= thirty_days_ago)
    ).one()

    # ── topic trends (real) ────────────────────────────────────
    # Find top 4 narrative labels by claim count
    top_narratives_rows = session.exec(
        select(Narrative.narrative_label, func.count(ClaimNarrative.claim_id))
        .join(ClaimNarrative, Narrative.narrative_id == ClaimNarrative.narrative_id)
        .group_by(Narrative.narrative_label)
        .order_by(func.count(ClaimNarrative.claim_id).desc())
        .limit(4)
    ).all()
    top_labels = [row[0] for row in top_narratives_rows]

    # Build time-series: claims per 2-week bucket per narrative label
    topic_trends: list[dict] = []
    if top_labels:
        rows = session.exec(
            select(
                Claim.created_at,
                Narrative.narrative_label,
            )
            .join(ClaimNarrative, Claim.claim_id == ClaimNarrative.claim_id)
            .join(Narrative, ClaimNarrative.narrative_id == Narrative.narrative_id)
            .where(
                Claim.created_at >= ninety_days_ago,
                Narrative.narrative_label.in_(top_labels),  # type: ignore[union-attr]
            )
        ).all()

        # Bucket by day
        buckets: dict[str, dict[str, int]] = defaultdict(
            lambda: {label: 0 for label in top_labels}
        )
        for created_at, label in rows:
            bucket_key = created_at.strftime("%Y-%m-%d")
            buckets[bucket_key][label] += 1

        for date_key in sorted(buckets):
            entry: dict = {"date": date_key, **buckets[date_key]}
            topic_trends.append(entry)

    # ── derived from real data ──────────────────────────────────
    verified_claims = session.exec(
        select(func.count())
        .select_from(Claim)
        .where(col(Claim.fact_check_status).ilike("verified"))
    ).one()

    disputed_claims = session.exec(
        select(func.count())
        .select_from(Claim)
        .where(col(Claim.fact_check_status).ilike("disputed"))
    ).one()

    checked_total = verified_claims + disputed_claims
    accuracy_pct = (
        round(verified_claims / checked_total * 100) if checked_total > 0 else 0
    )

    high_risk_alerts = session.exec(
        select(func.count())
        .select_from(Narrative)
        .where(Narrative.narrative_risk_score >= 7.0)
    ).one()

    if high_risk_alerts == 0:
        status_label = "All clear"
    elif high_risk_alerts <= 3:
        status_label = "Monitoring"
    else:
        status_label = "Action needed"

    avg_confidence = session.exec(
        select(func.avg(Claim.llm_confidence)).where(
            Claim.llm_confidence.is_not(None)  # type: ignore[union-attr]
        )
    ).one()
    confidence_score_pct = round(float(avg_confidence) * 100) if avg_confidence else 0

    return ExecutiveOverviewResponse(
        total_videos_scoped=total_videos,
        active_narratives=total_narratives,
        verified_claims=verified_claims,
        high_risk_alerts=high_risk_alerts,
        overview_metrics_meta={
            "total_videos_scoped": {
                "delta_pct": delta_pct,
                "delta_period_label": "vs last month",
            },
            "active_narratives": {
                "delta_new": new_narratives,
                "delta_period_label": "new",
            },
            "verified_claims": {
                "accuracy_pct": accuracy_pct,
                "accuracy_label": "accuracy",
            },
            "high_risk_alerts": {
                "status_label": status_label,
            },
        },
        topic_trends_meta={
            "window_days": 90,
            "confidence_score_pct": confidence_score_pct,
        },
        topic_trends=topic_trends,
    )


@router.get("/overview/topic-clusters", response_model=TopicClustersResponse)
def topic_clusters(session: Session = Depends(get_session)):
    rows = session.exec(
        select(
            Narrative.narrative_category,
            func.count(ClaimNarrative.claim_id).label("size"),
        )
        .outerjoin(
            ClaimNarrative,
            Narrative.narrative_id == ClaimNarrative.narrative_id,
        )
        .group_by(Narrative.narrative_category)
        .order_by(func.count(ClaimNarrative.claim_id).desc())
    ).all()

    clusters: list[dict] = []
    total = len(rows)
    for idx, (category, size) in enumerate(rows):
        # Distribute points in a grid-like pattern for visual spread
        angle = (idx / max(total, 1)) * 3.14159 * 2
        radius = 0.25 + (idx % 3) * 0.1
        clusters.append(
            {
                "label": category or "Uncategorized",
                "size": size,
                "x": round(0.5 + radius * math.cos(angle), 2),
                "y": round(0.5 + radius * math.sin(angle), 2),
            }
        )

    return TopicClustersResponse(clusters=clusters)
