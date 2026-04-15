from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlmodel import Session, select, func

from app.core.database import get_session
from app.models.video import Video
from app.models.claim import Claim
from app.models.narrative import Narrative
from app.models.claim_narrative import ClaimNarrative
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

    total_claims = session.exec(select(func.count()).select_from(Claim)).one()

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

        # Bucket into 2-week intervals
        buckets: dict[str, dict[str, int]] = defaultdict(
            lambda: {label: 0 for label in top_labels}
        )
        for created_at, label in rows:
            bucket_start = ninety_days_ago + timedelta(
                weeks=2 * ((created_at - ninety_days_ago).days // 14)
            )
            bucket_key = bucket_start.strftime("%Y-%m-%d")
            buckets[bucket_key][label] += 1

        for date_key in sorted(buckets):
            entry: dict = {"date": date_key, **buckets[date_key]}
            topic_trends.append(entry)

    # ── placeholder data (DB gaps) ─────────────────────────────
    # TODO: replace with verified-only count once claims.verification_status column exists
    verified_claims = total_claims

    # TODO: compute from real verification data
    accuracy_pct = 87

    # TODO: query alerts table once it exists
    high_risk_alerts = 0

    # TODO: derive from alerts review status
    status_label = "No data yet"

    # TODO: compute from real claim/narrative confidence scores
    confidence_score_pct = 92

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


# TODO: replace with real clustering from narrative embeddings / semantic similarity
@router.get("/overview/topic-clusters", response_model=TopicClustersResponse)
def topic_clusters():
    return TopicClustersResponse(
        clusters=[
            {"label": "Health & Wellness", "size": 42, "x": 0.32, "y": 0.55},
            {"label": "AI & Technology", "size": 28, "x": 0.51, "y": 0.41},
            {"label": "Crypto & Finance", "size": 34, "x": 0.63, "y": 0.66},
            {"label": "Nutrition", "size": 18, "x": 0.76, "y": 0.48},
        ],
    )
