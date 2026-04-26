from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlmodel import Session, select
from supabase import create_client

from app.core.config import settings
from app.core.database import engine
from app.core.email import send_email
from app.models.email_subscriber import EmailSubscriber
from app.models.notification_log import NotificationLog

router = APIRouter()
logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 7.0
MEDIUM_RISK_THRESHOLD = 4.0


def _build_alert_html(narratives: list[dict], risk_label: str = "High") -> str:
    """Build an HTML email body summarizing risk narratives."""
    rows = ""
    td = 'style="padding:8px;border:1px solid #ddd"'
    td_risk = 'style="padding:8px;border:1px solid #ddd;color:#dc2626;font-weight:bold"'
    for n in narratives:
        label = n.get("narrative_label", "Unknown")
        cat = n.get("narrative_category", "")
        score = n.get("narrative_risk_score", 0)
        desc = n.get("narrative_description", "")
        rows += (
            f"<tr>"
            f"<td {td}>{label}</td>"
            f"<td {td}>{cat}</td>"
            f"<td {td_risk}>{score}</td>"
            f"<td {td}>{desc}</td>"
            f"</tr>"
        )

    return (
        "<html><body>"
        f'<h2 style="color:#dc2626">New {risk_label}-Risk Narrative Alerts</h2>'
        f"<p>{len(narratives)} new {risk_label.lower()}-risk narrative(s) detected.</p>"
        '<table style="border-collapse:collapse;width:100%">'
        '<tr style="background:#f3f4f6">'
        '<th style="padding:8px;border:1px solid #ddd;text-align:left">Narrative</th>'
        '<th style="padding:8px;border:1px solid #ddd;text-align:left">Category</th>'
        '<th style="padding:8px;border:1px solid #ddd;text-align:left">Risk Score</th>'
        '<th style="padding:8px;border:1px solid #ddd;text-align:left">Description</th>'
        "</tr>"
        f"{rows}"
        "</table>"
        "</body></html>"
    )


def _send_notifications_background() -> None:
    """Background task: fetch new narratives and send per-subscriber digest emails."""
    try:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
            logger.error("Supabase not configured")
            return

        sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        # Fetch all narratives at or above the medium-risk threshold in one query
        raw_narratives = (
            sb.table("narratives")
            .select(
                "narrative_id, narrative_label, narrative_category, "
                "narrative_risk_score, narrative_description"
            )
            .gte("narrative_risk_score", MEDIUM_RISK_THRESHOLD)
            .execute()
            .data
            or []
        )

        if not raw_narratives:
            logger.info("No medium-or-high-risk narratives found")
            return

        # Classify each narrative as "High" or "Medium"
        for n in raw_narratives:
            score = float(n.get("narrative_risk_score") or 0)
            n["risk_level"] = "High" if score >= HIGH_RISK_THRESHOLD else "Medium"

        with Session(engine) as session:
            # Fetch all active subscribers
            subscribers = session.exec(
                select(EmailSubscriber).where(EmailSubscriber.is_active == True)  # noqa: E712
            ).all()

            if not subscribers:
                logger.info("No active subscribers")
                return

            # Bulk-query notification_log for already-sent (narrative_id, email) pairs
            narrative_ids = [
                uuid.UUID(n["narrative_id"])
                for n in raw_narratives
                if n.get("narrative_id")
            ]

            already_sent: set[tuple[str, str]] = set()
            if narrative_ids:
                already_sent_rows = session.exec(
                    select(NotificationLog).where(
                        NotificationLog.narrative_id.in_(narrative_ids)  # type: ignore[union-attr]
                    )
                ).all()
                already_sent = {
                    (str(row.narrative_id), row.recipient_email)
                    for row in already_sent_rows
                }

            sent_count = 0
            for subscriber in subscribers:
                # Filter narratives for this subscriber based on preferences and dedup
                to_send: list[dict] = []
                for n in raw_narratives:
                    risk_level = n["risk_level"]

                    if risk_level == "High" and not subscriber.notify_high_risk:
                        continue
                    if risk_level == "Medium" and not subscriber.notify_medium_risk:
                        continue

                    if (n["narrative_id"], subscriber.email) in already_sent:
                        continue

                    to_send.append(n)

                if not to_send:
                    continue

                # Build subject line based on risk levels present
                levels_present = {n["risk_level"] for n in to_send}
                if levels_present == {"High", "Medium"}:
                    risk_label = "High & Medium"
                else:
                    risk_label = next(iter(levels_present))

                subject = (
                    f"[Alert] {len(to_send)} {risk_label}-Risk Narrative(s) Detected"
                )
                html = _build_alert_html(to_send, risk_label=risk_label)

                try:
                    send_email(
                        to_email=subscriber.email,
                        to_name=subscriber.email,
                        subject=subject,
                        html_content=html,
                        sender_email=settings.ALERT_SENDER_EMAIL,
                        sender_name=settings.ALERT_SENDER_NAME,
                    )

                    # Log each sent narrative only after a confirmed successful send
                    for n in to_send:
                        session.add(
                            NotificationLog(
                                narrative_id=uuid.UUID(n["narrative_id"]),
                                recipient_email=subscriber.email,
                            )
                        )
                    session.commit()
                    sent_count += 1

                except Exception as exc:
                    logger.error(
                        "Failed to send alert email to %s: %s", subscriber.email, exc
                    )
                    session.rollback()

        logger.info(
            "Sent digest to %d/%d subscribers (%d narratives in scope)",
            sent_count,
            len(subscribers),
            len(raw_narratives),
        )

    except Exception as exc:
        logger.error("Error in background notification task: %s", exc)


@router.post("/alerts/notify")
def send_alert_notifications(background_tasks: BackgroundTasks):
    """Called by the database webhook to trigger email notifications.

    Returns immediately and sends emails in the background.
    """
    if not settings.BREVO_API_KEY:
        raise HTTPException(status_code=503, detail="BREVO_API_KEY not configured")

    background_tasks.add_task(_send_notifications_background)
    return {"ok": True, "detail": "Notification task queued"}
