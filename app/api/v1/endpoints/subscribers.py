from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select

from app.core.database import get_session
from app.models.email_subscriber import EmailSubscriber

router = APIRouter()


class SubscribeRequest(BaseModel):
    email: EmailStr
    notify_high_risk: bool = True
    notify_medium_risk: bool = False


class UnsubscribeRequest(BaseModel):
    email: EmailStr


@router.post("/alerts/subscribe")
def subscribe(body: SubscribeRequest, session: Session = Depends(get_session)):
    """Subscribe an email address to narrative alerts."""
    existing = session.exec(
        select(EmailSubscriber).where(EmailSubscriber.email == body.email)
    ).first()
    if existing:
        changed = not existing.is_active
        if existing.is_active:
            if (
                existing.notify_high_risk == body.notify_high_risk
                and existing.notify_medium_risk == body.notify_medium_risk
            ):
                return {
                    "ok": True,
                    "detail": "Already subscribed with these preferences",
                }
        existing.is_active = True
        existing.notify_high_risk = body.notify_high_risk
        existing.notify_medium_risk = body.notify_medium_risk
        session.add(existing)
        session.commit()
        return {
            "ok": True,
            "detail": "Re-subscribed" if changed else "Subscription updated",
        }
    subscriber = EmailSubscriber(
        email=body.email,
        notify_high_risk=body.notify_high_risk,
        notify_medium_risk=body.notify_medium_risk,
    )
    session.add(subscriber)
    session.commit()
    return {"ok": True, "detail": "Subscribed successfully"}


@router.post("/alerts/unsubscribe")
def unsubscribe(body: UnsubscribeRequest, session: Session = Depends(get_session)):
    """Unsubscribe an email address from alerts."""
    existing = session.exec(
        select(EmailSubscriber).where(EmailSubscriber.email == body.email)
    ).first()
    if not existing or not existing.is_active:
        return {"ok": False, "detail": "This email is not currently subscribed"}
    existing.is_active = False
    session.add(existing)
    session.commit()
    return {"ok": True, "detail": "Unsubscribed successfully"}


@router.get("/alerts/subscribers")
def check_subscription(
    email: str = Query(...), session: Session = Depends(get_session)
):
    """Check whether an email address is subscribed to alerts."""
    existing = session.exec(
        select(EmailSubscriber).where(EmailSubscriber.email == email)
    ).first()
    if existing and existing.is_active:
        return {
            "ok": True,
            "subscribed": True,
            "notify_high_risk": existing.notify_high_risk,
            "notify_medium_risk": existing.notify_medium_risk,
        }
    return {
        "ok": True,
        "subscribed": False,
        "notify_high_risk": True,
        "notify_medium_risk": False,
    }
