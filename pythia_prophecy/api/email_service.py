"""
Email service for sending confirmation and notification emails
In production, integrate with SendGrid, SES, or similar service
"""
import smtplib
import os
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from time import sleep

from .logging_config import get_logger

logger = get_logger("email")

# Email configuration from environment
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@pythia.example.com")
FROM_NAME = os.getenv("FROM_NAME", "Pythia")

# Frontend URL for email links
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Dev mode - print emails instead of sending
EMAIL_DEV_MODE = os.getenv("EMAIL_DEV_MODE", "true").lower() == "true"

# Retry configuration
EMAIL_RETRY_ATTEMPTS = int(os.getenv("EMAIL_RETRY_ATTEMPTS", "3"))
EMAIL_RETRY_DELAY = float(os.getenv("EMAIL_RETRY_DELAY", "5"))  # seconds

# Environment validation
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
if ENVIRONMENT == "production" and EMAIL_DEV_MODE:
    logger.warning("EMAIL_DEV_MODE is enabled in production! Emails will not be sent.")

if ENVIRONMENT == "production" and not SMTP_USER:
    logger.warning("SMTP_USER not configured in production. Emails will fail to send.")

if ENVIRONMENT == "production" and not SMTP_PASSWORD:
    logger.warning("SMTP_PASSWORD not configured in production. Emails will fail to send.")


def send_verification_email(to_email: str, first_name: str, token: str) -> bool:
    """Send email verification code to user."""
    subject = "Your Pythia Verification Code"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; padding: 20px 0; }}
            .logo {{ font-size: 32px; font-weight: bold; color: #6366f1; }}
            .content {{ background: #f9fafb; border-radius: 8px; padding: 30px; margin: 20px 0; }}
            .code-box {{ background: white; border: 2px solid #6366f1; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0; }}
            .code {{ font-size: 36px; font-weight: bold; color: #6366f1; letter-spacing: 2px; font-family: monospace; }}
            .footer {{ text-align: center; color: #6b7280; font-size: 14px; padding: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Pythia</div>
            </div>
            <div class="content">
                <h2>Welcome, {first_name}!</h2>
                <p>Thanks for signing up for Pythia. Use the verification code below to activate your account:</p>
                <div class="code-box">
                    <div class="code">{token}</div>
                </div>
                <p style="text-align: center; color: #6b7280; font-size: 14px;">
                    This code expires in 24 hours
                </p>
                <p style="color: #6b7280; font-size: 14px;">
                    If you didn't create an account, you can safely ignore this email.
                </p>
            </div>
            <div class="footer">
                <p>&copy; 2025 Pythia. All rights reserved.</p>
                <p>This is an automated message. Please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    Welcome to Pythia, {first_name}!

    Thanks for signing up. Use this code to verify your email:

    {token}

    This code expires in 24 hours.

    If you didn't create an account, you can safely ignore this email.

    - The Pythia Team
    """

    return _send_email(to_email, subject, text_content, html_content)


def send_welcome_email(to_email: str, first_name: str, tier: str) -> bool:
    """Send welcome email after verification."""
    subject = "Welcome to Pythia - Your account is ready!"

    tier_name = tier.capitalize()
    dashboard_url = f"{FRONTEND_URL}/dashboard"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; padding: 20px 0; }}
            .logo {{ font-size: 32px; font-weight: bold; color: #6366f1; }}
            .content {{ background: #f9fafb; border-radius: 8px; padding: 30px; margin: 20px 0; }}
            .button {{ display: inline-block; background: #6366f1; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: 500; }}
            .tier-badge {{ display: inline-block; background: #6366f1; color: white; padding: 4px 12px; border-radius: 20px; font-size: 14px; }}
            .footer {{ text-align: center; color: #6b7280; font-size: 14px; padding: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Pythia</div>
            </div>
            <div class="content">
                <h2>Your account is ready, {first_name}!</h2>
                <p>Your email has been verified and your <span class="tier-badge">{tier_name}</span> account is now active.</p>
                <p>You can now access AI-powered stock predictions and start making more informed decisions.</p>
                <p style="text-align: center; margin: 30px 0;">
                    <a href="{dashboard_url}" class="button">Go to Dashboard</a>
                </p>
            </div>
            <div class="footer">
                <p>&copy; 2025 Pythia. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    Your Pythia account is ready, {first_name}!

    Your email has been verified and your {tier_name} account is now active.

    Go to your dashboard: {dashboard_url}

    - The Pythia Team
    """

    return _send_email(to_email, subject, text_content, html_content)


def send_password_reset_email(to_email: str, first_name: str, token: str) -> bool:
    """Send password reset email."""
    subject = "Reset Your Pythia Password"

    reset_url = f"{FRONTEND_URL}/reset-password?token={token}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; padding: 20px 0; }}
            .logo {{ font-size: 32px; font-weight: bold; color: #6366f1; }}
            .content {{ background: #f9fafb; border-radius: 8px; padding: 30px; margin: 20px 0; }}
            .button {{ display: inline-block; background: #6366f1; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: 500; }}
            .footer {{ text-align: center; color: #6b7280; font-size: 14px; padding: 20px 0; }}
            .warning {{ background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 15px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Pythia</div>
            </div>
            <div class="content">
                <h2>Password Reset Request</h2>
                <p>Hi {first_name},</p>
                <p>We received a request to reset your Pythia password. Click the button below to set a new password:</p>
                <p style="text-align: center; margin: 30px 0;">
                    <a href="{reset_url}" class="button">Reset Password</a>
                </p>
                <p style="color: #6b7280; font-size: 14px;">
                    Or copy this link: <code>{reset_url}</code>
                </p>
                <div class="warning">
                    <strong>⚠️ Security Notice:</strong> This link expires in 1 hour. If you didn't request a password reset, please ignore this email or contact support.
                </div>
            </div>
            <div class="footer">
                <p>&copy; 2025 Pythia. All rights reserved.</p>
                <p>This is an automated message. Please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    Password Reset Request

    Hi {first_name},

    We received a request to reset your Pythia password. Use this link to set a new password:

    {reset_url}

    This link expires in 1 hour.

    If you didn't request a password reset, please ignore this email.

    - The Pythia Team
    """

    return _send_email(to_email, subject, text_content, html_content)


def _send_email(
    to_email: str,
    subject: str,
    text_content: str,
    html_content: Optional[str] = None,
) -> bool:
    """Send an email using SMTP with retry logic or log in dev mode."""
    if EMAIL_DEV_MODE:
        logger.info(f"[DEV MODE] Email to {to_email}: {subject}")
        logger.debug(f"[DEV MODE] Email text content:\n{text_content}")
        if html_content:
            logger.debug(f"[DEV MODE] Email HTML content:\n{html_content}")
        return True

    # Attempt to send with retries
    for attempt in range(1, EMAIL_RETRY_ATTEMPTS + 1):
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
            msg["To"] = to_email

            msg.attach(MIMEText(text_content, "plain"))
            if html_content:
                msg.attach(MIMEText(html_content, "html"))

            # Use SSL/TLS for port 465, STARTTLS for port 587
            if SMTP_PORT == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                    if SMTP_USER and SMTP_PASSWORD:
                        server.login(SMTP_USER, SMTP_PASSWORD)
                    server.sendmail(FROM_EMAIL, to_email, msg.as_string())
            else:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    if SMTP_USER and SMTP_PASSWORD:
                        server.starttls()
                        server.login(SMTP_USER, SMTP_PASSWORD)
                    server.sendmail(FROM_EMAIL, to_email, msg.as_string())

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
