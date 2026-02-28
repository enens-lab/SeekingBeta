"""
Email service for sending confirmation and notification emails
In production, integrate with SendGrid, SES, or similar service
"""
import smtplib
import os
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid
from html import escape
from typing import Optional
from time import sleep
from urllib.parse import quote_plus

from .logging_config import get_logger
from .auth import create_access_token

logger = get_logger("email")

# Email configuration from environment
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@seekingbeta.ai")
FROM_NAME = os.getenv("FROM_NAME", "SeekingBeta")
APP_NAME = os.getenv("APP_NAME", "SeekingBeta")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@seekingbeta.ai")
REPLY_TO_EMAIL = os.getenv("REPLY_TO_EMAIL", SUPPORT_EMAIL)

# Frontend URL for email links
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Dev mode - print emails instead of sending
EMAIL_DEV_MODE = os.getenv("EMAIL_DEV_MODE", "true").lower() == "true"

# Retry configuration
EMAIL_RETRY_ATTEMPTS = int(os.getenv("EMAIL_RETRY_ATTEMPTS", "3"))
EMAIL_RETRY_DELAY = float(os.getenv("EMAIL_RETRY_DELAY", "5"))  # seconds
EMAIL_SUPPRESSION_ENABLED = (
    os.getenv("EMAIL_SUPPRESSION_ENABLED", "true").lower() == "true"
)

# Environment validation
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
if ENVIRONMENT == "production" and EMAIL_DEV_MODE:
    logger.warning("EMAIL_DEV_MODE is enabled in production! Emails will not be sent.")

if ENVIRONMENT == "production" and not SMTP_USER:
    logger.warning("SMTP_USER not configured in production. Emails will fail to send.")

if ENVIRONMENT == "production" and not SMTP_PASSWORD:
    logger.warning("SMTP_PASSWORD not configured in production. Emails will fail to send.")


def _safe_name(name: str) -> str:
    clean = (name or "").strip()
    if not clean:
        return "there"
    return escape(clean)


def _render_email_html(
    title: str,
    body_html: str,
    cta_label: Optional[str] = None,
    cta_url: Optional[str] = None,
) -> str:
    cta_block = ""
    if cta_label and cta_url:
        cta_block = (
            '<p style="text-align: center; margin: 24px 0;">'
            f'<a href="{cta_url}" class="button">{cta_label}</a>'
            "</p>"
        )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #0f172a; background: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            .header {{ text-align: center; padding: 12px 0 20px; }}
            .logo {{ font-size: 30px; font-weight: 700; color: #008f7a; }}
            .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 28px; }}
            .button {{ display: inline-block; background: #008f7a; color: #ffffff !important; padding: 12px 24px; text-decoration: none; border-radius: 8px; font-weight: 600; }}
            .muted {{ color: #64748b; font-size: 14px; }}
            .footer {{ text-align: center; color: #64748b; font-size: 13px; padding: 16px 0 0; }}
            .code-box {{ background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; text-align: center; margin: 16px 0; }}
            .code {{ font-size: 28px; font-weight: 700; color: #0f766e; letter-spacing: 2px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
            .notice {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 12px; margin: 16px 0; color: #9a3412; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">{APP_NAME}</div>
            </div>
            <div class="card">
                <h2>{title}</h2>
                {body_html}
                {cta_block}
            </div>
            <div class="footer">
                <p>&copy; 2026 {APP_NAME}. All rights reserved.</p>
                <p>This is an automated transactional message. Please do not reply directly.</p>
            </div>
        </div>
    </body>
    </html>
    """


def send_verification_email(to_email: str, first_name: str, token: str) -> bool:
    """Send email verification code to user."""
    subject = f"Verify your {APP_NAME} account"
    verify_url = f"{FRONTEND_URL.rstrip('/')}/verify-email?token={quote_plus(token)}&email={quote_plus(to_email)}"
    safe_name = _safe_name(first_name)
    safe_token = escape(token)
    safe_url = escape(verify_url)

    body_html = f"""
    <p>Hi {safe_name},</p>
    <p>Thanks for creating your {APP_NAME} account. Please verify your email address to continue.</p>
    <p>If the button does not work, enter this code in the app:</p>
    <div class="code-box">
        <div class="code">{safe_token}</div>
    </div>
    <p class="muted">Verification link: <a href="{safe_url}">{safe_url}</a></p>
    <p class="muted">This code expires in 24 hours.</p>
    <p class="muted">If you did not create this account, you can ignore this email.</p>
    """
    html_content = _render_email_html(
        title="Verify your email",
        body_html=body_html,
        cta_label="Verify Email",
        cta_url=safe_url,
    )

    text_content = f"""
    Welcome to {APP_NAME}, {first_name}!

    Verify your account using this link:
    {verify_url}

    Or enter this code in the app:

    {token}

    This code expires in 24 hours.

    If you didn't create an account, you can safely ignore this email.

    - The {APP_NAME} Team
    """.strip()

    if EMAIL_DEV_MODE:
        logger.info(f"[DEV MODE] Verification code for {to_email}: {token}")
        logger.info(f"[DEV MODE] Verification URL for {to_email}: {verify_url}")

    return _send_email(
        to_email,
        subject,
        text_content,
        html_content,
        category="transactional",
    )


def send_welcome_email(to_email: str, first_name: str, tier: str) -> bool:
    """Send welcome email after verification."""
    subject = f"Welcome to {APP_NAME} - your account is ready"

    tier_name = tier.capitalize()
    dashboard_url = f"{FRONTEND_URL.rstrip('/')}/dashboard"
    safe_name = _safe_name(first_name)
    safe_tier = escape(tier_name)
    safe_url = escape(dashboard_url)

    body_html = f"""
    <p>Hi {safe_name},</p>
    <p>Your email has been verified and your <strong>{safe_tier}</strong> account is active.</p>
    <p>You can now access stock predictions and manage your watchlist from the dashboard.</p>
    """
    html_content = _render_email_html(
        title="Welcome aboard",
        body_html=body_html,
        cta_label="Open Dashboard",
        cta_url=safe_url,
    )

    text_content = f"""
    Your {APP_NAME} account is ready, {first_name}!

    Your email has been verified and your {tier_name} account is now active.

    Go to your dashboard: {dashboard_url}

    - The {APP_NAME} Team
    """.strip()

    return _send_email(
        to_email,
        subject,
        text_content,
        html_content,
        category="transactional",
    )


def send_password_reset_email(to_email: str, first_name: str, token: str) -> bool:
    """Send password reset email."""
    subject = f"Reset your {APP_NAME} password"

    reset_url = f"{FRONTEND_URL.rstrip('/')}/reset-password?token={quote_plus(token)}"
    safe_name = _safe_name(first_name)
    safe_url = escape(reset_url)

    body_html = f"""
    <p>Hi {safe_name},</p>
    <p>We received a request to reset your password.</p>
    <p class="muted">If you did not request this, you can ignore this email.</p>
    <div class="notice"><strong>Security notice:</strong> This reset link expires in 1 hour.</div>
    <p class="muted">Direct link: <a href="{safe_url}">{safe_url}</a></p>
    """
    html_content = _render_email_html(
        title="Password reset request",
        body_html=body_html,
        cta_label="Reset Password",
        cta_url=safe_url,
    )

    text_content = f"""
    Password Reset Request

    Hi {first_name},

    We received a request to reset your {APP_NAME} password. Use this link to set a new password:

    {reset_url}

    This link expires in 1 hour.

    If you didn't request a password reset, please ignore this email.

    - The {APP_NAME} Team
    """.strip()

    return _send_email(
        to_email,
        subject,
        text_content,
        html_content,
        category="transactional",
    )


def build_unsubscribe_url(user_id: str, email: str) -> str:
    """Build one-click unsubscribe URL for future marketing emails."""
    token = create_access_token(user_id, email, token_type="unsubscribe")
    return f"{FRONTEND_URL.rstrip('/')}/api/email/unsubscribe?token={quote_plus(token)}"


def _send_email(
    to_email: str,
    subject: str,
    text_content: str,
    html_content: Optional[str] = None,
    category: str = "transactional",
    unsubscribe_url: Optional[str] = None,
) -> bool:
    """Send an email using SMTP with retry logic or log in dev mode."""
    if EMAIL_DEV_MODE:
        logger.info(f"[DEV MODE] Email to {to_email}: {subject}")
        logger.debug(f"[DEV MODE] Email text content:\n{text_content}")
        if html_content:
            logger.debug(f"[DEV MODE] Email HTML content:\n{html_content}")
        return True

    if EMAIL_SUPPRESSION_ENABLED:
        try:
            from .compliance_store import get_active_suppression

            suppression = get_active_suppression(to_email)
        except Exception as exc:
            logger.error(f"Suppression lookup failed for {to_email}: {exc}")
            suppression = None

        if suppression:
            logger.warning(
                "Blocked email send to suppressed recipient %s (reason=%s, source=%s)",
                to_email,
                suppression.get("reason"),
                suppression.get("source"),
            )
            return False

    # Build message once so retries reuse the same payload/message-id.
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Reply-To"] = REPLY_TO_EMAIL or FROM_EMAIL
    msg["Date"] = formatdate(localtime=False)
    from_domain = FROM_EMAIL.split("@", 1)[-1] if "@" in FROM_EMAIL else None
    message_id = make_msgid(domain=from_domain)
    msg["Message-ID"] = message_id
    msg["Auto-Submitted"] = "auto-generated"
    msg["X-Auto-Response-Suppress"] = "OOF, AutoReply"
    msg["X-Entity-Ref-ID"] = message_id.strip("<>")
    msg["X-Email-Category"] = category
    if unsubscribe_url:
        msg["List-Unsubscribe"] = f"<{unsubscribe_url}>"

    msg.attach(MIMEText(text_content, "plain"))
    if html_content:
        msg.attach(MIMEText(html_content, "html"))

    msg_str = msg.as_string()

    # Attempt to send with retries
    for attempt in range(1, EMAIL_RETRY_ATTEMPTS + 1):
        try:

            # Use SSL/TLS for port 465, STARTTLS for port 587
            if SMTP_PORT == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                    if SMTP_USER and SMTP_PASSWORD:
                        server.login(SMTP_USER, SMTP_PASSWORD)
                    server.sendmail(FROM_EMAIL, to_email, msg_str)
            else:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    if SMTP_USER and SMTP_PASSWORD:
                        server.starttls()
                        server.login(SMTP_USER, SMTP_PASSWORD)
                    server.sendmail(FROM_EMAIL, to_email, msg_str)

            logger.info(f"Email sent to {to_email}: {subject}")
            return True

        except Exception as e:
            if attempt < EMAIL_RETRY_ATTEMPTS:
                delay = EMAIL_RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"Failed to send email to {to_email} (attempt {attempt}/{EMAIL_RETRY_ATTEMPTS}). "
                    f"Retrying in {delay:.1f}s: {e}"
                )
                sleep(delay)
            else:
                logger.error(
                    f"Failed to send email to {to_email} after {EMAIL_RETRY_ATTEMPTS} attempts: {e}"
                )
                return False

    return False
