from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from supabase import create_client

from app.core.config import settings
from app.core.database import get_session
from app.core.email import send_email
from app.models.email_subscriber import EmailSubscriber

router = APIRouter()
logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 1.0  # TODO: restore to 7.0 after testing


def _build_alert_html(narratives: list[dict]) -> str:
    """Build an HTML email body summarizing high-risk narratives."""
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
        '<h2 style="color:#dc2626">New High-Risk Narrative Alerts</h2>'
        f"<p>{len(narratives)} new high-risk narrative(s) detected.</p>"
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


@router.post("/alerts/notify")
def send_alert_notifications(session: Session = Depends(get_session)):
    """Called by the pipeline after a run to email high-risk narratives."""
    if not settings.BREVO_API_KEY:
        raise HTTPException(status_code=503, detail="BREVO_API_KEY not configured")

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    # Fetch all high-risk narratives
    narratives = (
        sb.table("narratives")
        .select(
            "narrative_id, narrative_label, narrative_category, "
            "narrative_risk_score, narrative_description"
        )
        .gte("narrative_risk_score", HIGH_RISK_THRESHOLD)
        .execute()
        .data
        or []
    )

    if not narratives:
        return {"ok": True, "detail": "No high-risk narratives found", "notified": 0}

    # Fetch active subscribers from the database
    subscribers = session.exec(
        select(EmailSubscriber).where(EmailSubscriber.is_active == True)  # noqa: E712
    ).all()
    recipients = [s.email for s in subscribers]

    if not recipients:
        return {"ok": True, "detail": "No active subscribers", "notified": 0}

    # Build a single digest email
    html = _build_alert_html(narratives)
    subject = f"[Alert] {len(narratives)} High-Risk Narrative(s) Detected"

    sent_count = 0
    errors = []

    for email_addr in recipients:
        try:
            send_email(
                to_email=email_addr,
                to_name=email_addr,
                subject=subject,
                html_content=html,
                sender_email=settings.ALERT_SENDER_EMAIL,
                sender_name=settings.ALERT_SENDER_NAME,
            )
            sent_count += 1
        except Exception as exc:
            logger.error("Failed to send alert email to %s: %s", email_addr, exc)
            errors.append({"email": email_addr, "error": str(exc)})

    return {
        "ok": sent_count > 0,
        "detail": f"Sent digest to {sent_count}/{len(recipients)} recipients",
        "notified": len(narratives),
        "errors": errors,
    }
