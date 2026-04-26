from brevo import Brevo
from brevo.transactional_emails import (
    SendTransacEmailRequestSender,
    SendTransacEmailRequestToItem,
)

from app.core.config import settings


def send_email(
    to_email: str,
    to_name: str,
    subject: str,
    html_content: str,
    sender_email: str = "noreply@yourdomain.com",
    sender_name: str = "YouTube Intelligence",
) -> str:
    """Send a transactional email via Brevo.

    Returns the message ID on success.
    """
    client = Brevo(api_key=settings.BREVO_API_KEY)

    result = client.transactional_emails.send_transac_email(
        subject=subject,
        html_content=html_content,
        sender=SendTransacEmailRequestSender(
            name=sender_name,
            email=sender_email,
        ),
        to=[
            SendTransacEmailRequestToItem(
                email=to_email,
                name=to_name,
            )
        ],
    )

    return result.message_id
