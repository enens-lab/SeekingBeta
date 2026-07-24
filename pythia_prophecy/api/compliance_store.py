"""Postgres-backed compliance storage for consent, preferences, and suppression."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from .logging_config import get_logger

logger = get_logger("compliance_store")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
EMAIL_SUPPRESSION_ENABLED = (
    os.getenv("EMAIL_SUPPRESSION_ENABLED", "true").lower() == "true"
)

PREFERENCES_TABLE = "user_compliance_preferences"
CONSENT_EVENTS_TABLE = "user_consent_events"
SUPPRESSION_TABLE = "email_suppression_list"
EMAIL_AUDIT_TABLE = "email_event_audit"

DEFAULT_PREFERENCES = {
    "theme": "light",
    "email_alerts_enabled": True,
    "daily_digest_enabled": False,
    "newsletter_enabled": False,
    "two_factor_enabled": False,
    "language": "en",
    "timezone": "UTC",
    "notifications_enabled": True,
}


def is_enabled() -> bool:
    return bool(DATABASE_URL)


def suppression_enabled() -> bool:
    return is_enabled() and EMAIL_SUPPRESSION_ENABLED


@contextmanager
def _get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def ensure_tables() -> bool:
    if not is_enabled():
        logger.warning("DATABASE_URL not set; compliance store disabled")
        return False

    sql_preferences = f"""
        CREATE TABLE IF NOT EXISTS {PREFERENCES_TABLE} (
            user_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            theme TEXT NOT NULL DEFAULT 'light',
            email_alerts_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            daily_digest_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            newsletter_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            language TEXT NOT NULL DEFAULT 'en',
            timezone TEXT NOT NULL DEFAULT 'UTC',
            notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    sql_consent_events = f"""
        CREATE TABLE IF NOT EXISTS {CONSENT_EVENTS_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            consent_type TEXT NOT NULL,
            granted BOOLEAN NOT NULL,
            ip_address TEXT NULL,
            user_agent TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    sql_suppression = f"""
        CREATE TABLE IF NOT EXISTS {SUPPRESSION_TABLE} (
            email TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            source TEXT NOT NULL,
            provider_event_type TEXT NULL,
            provider_message_id TEXT NULL,
            raw_event JSONB NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    sql_email_audit = f"""
        CREATE TABLE IF NOT EXISTS {EMAIL_AUDIT_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            provider TEXT NOT NULL,
            event_type TEXT NOT NULL,
            email TEXT NULL,
            provider_message_id TEXT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    sql_indexes = [
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{PREFERENCES_TABLE}_email
        ON {PREFERENCES_TABLE}(email)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{CONSENT_EVENTS_TABLE}_user_id
        ON {CONSENT_EVENTS_TABLE}(user_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{CONSENT_EVENTS_TABLE}_email
        ON {CONSENT_EVENTS_TABLE}(email)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{SUPPRESSION_TABLE}_is_active
        ON {SUPPRESSION_TABLE}(is_active)
        """,
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{EMAIL_AUDIT_TABLE}_idem
        ON {EMAIL_AUDIT_TABLE}(provider, event_type, COALESCE(email, ''), COALESCE(provider_message_id, ''))
        """,
    ]

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_preferences)
            cur.execute(sql_consent_events)
            cur.execute(sql_suppression)
            cur.execute(sql_email_audit)
            for statement in sql_indexes:
                cur.execute(statement)
        conn.commit()

    logger.info("Compliance tables ready")
    return True


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _row_to_preferences(row: dict[str, Any], user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "theme": row["theme"],
        "email_alerts_enabled": bool(row["email_alerts_enabled"]),
        "daily_digest_enabled": bool(row["daily_digest_enabled"]),
        "newsletter_enabled": bool(row["newsletter_enabled"]),
        "two_factor_enabled": bool(row["two_factor_enabled"]),
        "language": row["language"],
        "timezone": row["timezone"],
        "notifications_enabled": bool(row["notifications_enabled"]),
        "updated_at": row["updated_at"],
    }


def get_or_create_preferences(user_id: str, email: str) -> dict[str, Any]:
    if not is_enabled():
        now = datetime.utcnow()
        return {
            "user_id": user_id,
            **DEFAULT_PREFERENCES,
            "updated_at": now,
        }

    safe_email = _normalize_email(email)
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                INSERT INTO {PREFERENCES_TABLE} (
                    user_id,
                    email,
                    theme,
                    email_alerts_enabled,
                    daily_digest_enabled,
                    newsletter_enabled,
                    two_factor_enabled,
                    language,
                    timezone,
                    notifications_enabled,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (user_id) DO NOTHING
                """,
                (
                    user_id,
                    safe_email,
                    DEFAULT_PREFERENCES["theme"],
                    DEFAULT_PREFERENCES["email_alerts_enabled"],
                    DEFAULT_PREFERENCES["daily_digest_enabled"],
                    DEFAULT_PREFERENCES["newsletter_enabled"],
                    DEFAULT_PREFERENCES["two_factor_enabled"],
                    DEFAULT_PREFERENCES["language"],
                    DEFAULT_PREFERENCES["timezone"],
                    DEFAULT_PREFERENCES["notifications_enabled"],
                ),
            )
            cur.execute(
                f"""
                SELECT theme, email_alerts_enabled, daily_digest_enabled, newsletter_enabled,
                       two_factor_enabled, language, timezone, notifications_enabled, updated_at
                FROM {PREFERENCES_TABLE}
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError(f"Could not create or load preferences for user_id={user_id}")
    return _row_to_preferences(dict(row), user_id)


def upsert_preferences(
    user_id: str,
    email: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    base = get_or_create_preferences(user_id, email)
    merged = {
        "theme": updates.get("theme", base["theme"]),
        "email_alerts_enabled": updates.get(
            "email_alerts_enabled", base["email_alerts_enabled"]
        ),
        "daily_digest_enabled": updates.get(
            "daily_digest_enabled", base["daily_digest_enabled"]
        ),
        "newsletter_enabled": updates.get(
            "newsletter_enabled", base["newsletter_enabled"]
        ),
        "two_factor_enabled": updates.get("two_factor_enabled", base["two_factor_enabled"]),
        "language": updates.get("language", base["language"]),
        "timezone": updates.get("timezone", base["timezone"]),
        "notifications_enabled": updates.get(
            "notifications_enabled", base["notifications_enabled"]
        ),
    }

    if not is_enabled():
        merged["user_id"] = user_id
        merged["updated_at"] = base["updated_at"]
        return merged

    safe_email = _normalize_email(email)
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                INSERT INTO {PREFERENCES_TABLE} (
                    user_id,
                    email,
                    theme,
                    email_alerts_enabled,
                    daily_digest_enabled,
                    newsletter_enabled,
                    two_factor_enabled,
                    language,
                    timezone,
                    notifications_enabled,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (user_id) DO UPDATE SET
                    email = EXCLUDED.email,
                    theme = EXCLUDED.theme,
                    email_alerts_enabled = EXCLUDED.email_alerts_enabled,
                    daily_digest_enabled = EXCLUDED.daily_digest_enabled,
                    newsletter_enabled = EXCLUDED.newsletter_enabled,
                    two_factor_enabled = EXCLUDED.two_factor_enabled,
                    language = EXCLUDED.language,
                    timezone = EXCLUDED.timezone,
                    notifications_enabled = EXCLUDED.notifications_enabled,
                    updated_at = NOW()
                RETURNING theme, email_alerts_enabled, daily_digest_enabled, newsletter_enabled,
                          two_factor_enabled, language, timezone, notifications_enabled, updated_at
                """,
                (
                    user_id,
                    safe_email,
                    merged["theme"],
                    merged["email_alerts_enabled"],
                    merged["daily_digest_enabled"],
                    merged["newsletter_enabled"],
                    merged["two_factor_enabled"],
                    merged["language"],
                    merged["timezone"],
                    merged["notifications_enabled"],
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError(f"Could not update preferences for user_id={user_id}")
    return _row_to_preferences(dict(row), user_id)


def list_daily_digest_recipients() -> list[dict[str, Any]]:
    """Users who opted into the Daily Brief. Suppression is enforced separately
    at send time (email_service._send_email), so this is just the opt-in list."""
    if not is_enabled():
        return []

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT user_id, email
                FROM {PREFERENCES_TABLE}
                WHERE daily_digest_enabled = TRUE
                ORDER BY email
                """
            )
            rows = cur.fetchall()
    return [{"user_id": row["user_id"], "email": row["email"]} for row in rows]


def set_daily_digest_opt_in(user_id: str, email: str, enabled: bool) -> dict[str, Any]:
    return upsert_preferences(
        user_id=user_id,
        email=email,
        updates={"daily_digest_enabled": bool(enabled)},
    )


def set_newsletter_opt_in(user_id: str, email: str, enabled: bool) -> dict[str, Any]:
    return upsert_preferences(
        user_id=user_id,
        email=email,
        updates={"newsletter_enabled": bool(enabled)},
    )


def disable_newsletter_by_email(email: str) -> bool:
    if not is_enabled():
        return False

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {PREFERENCES_TABLE}
                SET newsletter_enabled = FALSE, updated_at = NOW()
                WHERE email = %s
                """,
                (_normalize_email(email),),
            )
            updated = cur.rowcount
        conn.commit()
    return updated > 0


def delete_user_records(user_id: str, email: str) -> dict[str, int]:
    """Delete user-linked preference and consent records."""
    if not is_enabled():
        return {"preferences_deleted": 0, "consent_events_deleted": 0}

    safe_email = _normalize_email(email)
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {PREFERENCES_TABLE} WHERE user_id = %s OR email = %s",
                (user_id, safe_email),
            )
            preferences_deleted = cur.rowcount
            cur.execute(
                f"DELETE FROM {CONSENT_EVENTS_TABLE} WHERE user_id = %s OR email = %s",
                (user_id, safe_email),
            )
            consent_events_deleted = cur.rowcount
        conn.commit()

    return {
        "preferences_deleted": int(preferences_deleted),
        "consent_events_deleted": int(consent_events_deleted),
    }


def record_consent_event(
    user_id: str,
    email: str,
    policy_version: str,
    consent_type: str,
    granted: bool,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    if not is_enabled():
        return

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {CONSENT_EVENTS_TABLE} (
                    user_id, email, policy_version, consent_type, granted, ip_address, user_agent, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    user_id,
                    _normalize_email(email),
                    policy_version.strip(),
                    consent_type.strip().lower(),
                    bool(granted),
                    ip_address,
                    user_agent,
                ),
            )
        conn.commit()


def audit_email_event(
    provider: str,
    event_type: str,
    payload: dict[str, Any],
    email: Optional[str] = None,
    provider_message_id: Optional[str] = None,
) -> bool:
    if not is_enabled():
        return False

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {EMAIL_AUDIT_TABLE} (
                    provider, event_type, email, provider_message_id, payload, created_at
                ) VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
                """,
                (
                    provider.strip().lower(),
                    event_type.strip().lower(),
                    _normalize_email(email) if email else None,
                    provider_message_id,
                    Json(payload),
                ),
            )
            inserted = cur.rowcount > 0
        conn.commit()
    return inserted


def upsert_suppression(
    email: str,
    reason: str,
    source: str,
    provider_event_type: Optional[str] = None,
    provider_message_id: Optional[str] = None,
    raw_event: Optional[dict[str, Any]] = None,
) -> None:
    if not suppression_enabled():
        return

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SUPPRESSION_TABLE} (
                    email, reason, source, provider_event_type, provider_message_id, raw_event, is_active, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW())
                ON CONFLICT (email) DO UPDATE SET
                    reason = EXCLUDED.reason,
                    source = EXCLUDED.source,
                    provider_event_type = EXCLUDED.provider_event_type,
                    provider_message_id = EXCLUDED.provider_message_id,
                    raw_event = EXCLUDED.raw_event,
                    is_active = TRUE,
                    updated_at = NOW()
                """,
                (
                    _normalize_email(email),
                    reason.strip(),
                    source.strip(),
                    provider_event_type,
                    provider_message_id,
                    Json(raw_event) if raw_event else None,
                ),
            )
        conn.commit()


def get_active_suppression(email: str) -> Optional[dict[str, Any]]:
    if not suppression_enabled():
        return None

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT email, reason, source, provider_event_type, provider_message_id, is_active, updated_at
                FROM {SUPPRESSION_TABLE}
                WHERE email = %s AND is_active = TRUE
                LIMIT 1
                """,
                (_normalize_email(email),),
            )
            row = cur.fetchone()

    return dict(row) if row else None
