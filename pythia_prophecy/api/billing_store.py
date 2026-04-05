"""Postgres-backed billing storage for Stripe customer/subscription state."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from .logging_config import get_logger

logger = get_logger("billing_store")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

BILLING_STATE_TABLE = "billing_customer_state"
WEBHOOK_EVENTS_TABLE = "billing_webhook_events"


DEFAULT_STATE = {
    "plan_tier": "free",
    "subscription_status": "none",
    "price_id": None,
    "stripe_subscription_id": None,
    "stripe_customer_id": None,
    "current_period_end": None,
    "cancel_at_period_end": False,
    "legacy_grace_expires_at": None,
    "apple_tier": None,
    "apple_product_id": None,
    "apple_transaction_id": None,
    "apple_original_transaction_id": None,
    "apple_app_account_token": None,
    "apple_subscription_status": None,
    "apple_expires_at": None,
    "apple_environment": None,
    "apple_last_verified_at": None,
    "source": "system",
}


def is_enabled() -> bool:
    return bool(DATABASE_URL)


@contextmanager
def _get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def ensure_tables() -> bool:
    if not is_enabled():
        logger.warning("DATABASE_URL not set; billing store disabled")
        return False

    sql_state = f"""
        CREATE TABLE IF NOT EXISTS {BILLING_STATE_TABLE} (
            user_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            stripe_customer_id TEXT UNIQUE NULL,
            plan_tier TEXT NOT NULL DEFAULT 'free',
            subscription_status TEXT NOT NULL DEFAULT 'none',
            price_id TEXT NULL,
            stripe_subscription_id TEXT NULL,
            current_period_end TIMESTAMPTZ NULL,
            cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
            legacy_grace_expires_at TIMESTAMPTZ NULL,
            apple_tier TEXT NULL,
            apple_product_id TEXT NULL,
            apple_transaction_id TEXT NULL,
            apple_original_transaction_id TEXT NULL,
            apple_app_account_token TEXT NULL,
            apple_subscription_status TEXT NULL,
            apple_expires_at TIMESTAMPTZ NULL,
            apple_environment TEXT NULL,
            apple_last_verified_at TIMESTAMPTZ NULL,
            source TEXT NOT NULL DEFAULT 'system',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    sql_webhook = f"""
        CREATE TABLE IF NOT EXISTS {WEBHOOK_EVENTS_TABLE} (
            stripe_event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            processed_at TIMESTAMPTZ NULL,
            processing_status TEXT NOT NULL DEFAULT 'processing',
            error_message TEXT NULL
        )
    """

    sql_indexes = [
        f"""
        CREATE INDEX IF NOT EXISTS idx_{BILLING_STATE_TABLE}_stripe_customer_id
        ON {BILLING_STATE_TABLE}(stripe_customer_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{BILLING_STATE_TABLE}_stripe_subscription_id
        ON {BILLING_STATE_TABLE}(stripe_subscription_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{BILLING_STATE_TABLE}_subscription_status
        ON {BILLING_STATE_TABLE}(subscription_status)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{BILLING_STATE_TABLE}_legacy_grace_expires_at
        ON {BILLING_STATE_TABLE}(legacy_grace_expires_at)
        """,
    ]

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_state)
            cur.execute(sql_webhook)
            alter_statements = [
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_tier TEXT NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_product_id TEXT NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_transaction_id TEXT NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_original_transaction_id TEXT NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_app_account_token TEXT NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_subscription_status TEXT NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_expires_at TIMESTAMPTZ NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_environment TEXT NULL",
                f"ALTER TABLE {BILLING_STATE_TABLE} ADD COLUMN IF NOT EXISTS apple_last_verified_at TIMESTAMPTZ NULL",
            ]
            for statement in alter_statements:
                cur.execute(statement)
            for statement in sql_indexes:
                cur.execute(statement)
        conn.commit()

    logger.info("Billing tables ready")
    return True


def get_state_by_user_id(user_id: str) -> Optional[dict[str, Any]]:
    if not is_enabled():
        return None
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT user_id, email, stripe_customer_id, plan_tier, subscription_status,
                       price_id, stripe_subscription_id, current_period_end,
                       cancel_at_period_end, legacy_grace_expires_at,
                       apple_tier, apple_product_id, apple_transaction_id,
                       apple_original_transaction_id, apple_app_account_token,
                       apple_subscription_status, apple_expires_at,
                       apple_environment, apple_last_verified_at, source,
                       created_at, updated_at
                FROM {BILLING_STATE_TABLE}
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def get_state_by_customer_id(stripe_customer_id: str) -> Optional[dict[str, Any]]:
    if not is_enabled() or not stripe_customer_id:
        return None
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT user_id, email, stripe_customer_id, plan_tier, subscription_status,
                       price_id, stripe_subscription_id, current_period_end,
                       cancel_at_period_end, legacy_grace_expires_at,
                       apple_tier, apple_product_id, apple_transaction_id,
                       apple_original_transaction_id, apple_app_account_token,
                       apple_subscription_status, apple_expires_at,
                       apple_environment, apple_last_verified_at, source,
                       created_at, updated_at
                FROM {BILLING_STATE_TABLE}
                WHERE stripe_customer_id = %s
                """,
                (stripe_customer_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def delete_state_by_user_id(user_id: str) -> bool:
    if not is_enabled():
        return False
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {BILLING_STATE_TABLE} WHERE user_id = %s",
                (user_id,),
            )
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def upsert_customer_state(
    user_id: str,
    email: str,
    updates: Optional[dict[str, Any]] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    if not is_enabled():
        raise RuntimeError("Billing store disabled")

    state = get_state_by_user_id(user_id) or {
        "user_id": user_id,
        "email": _normalize_email(email),
        **DEFAULT_STATE,
    }

    merged = dict(state)
    merged["email"] = _normalize_email(email)
    if updates:
        merged.update(updates)
    if source is not None:
        merged["source"] = source

    merged["current_period_end"] = _as_utc(merged.get("current_period_end"))
    merged["legacy_grace_expires_at"] = _as_utc(merged.get("legacy_grace_expires_at"))
    merged["apple_expires_at"] = _as_utc(merged.get("apple_expires_at"))
    merged["apple_last_verified_at"] = _as_utc(merged.get("apple_last_verified_at"))
    merged["cancel_at_period_end"] = bool(merged.get("cancel_at_period_end", False))

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                INSERT INTO {BILLING_STATE_TABLE} (
                    user_id,
                    email,
                    stripe_customer_id,
                    plan_tier,
                    subscription_status,
                    price_id,
                    stripe_subscription_id,
                    current_period_end,
                    cancel_at_period_end,
                    legacy_grace_expires_at,
                    apple_tier,
                    apple_product_id,
                    apple_transaction_id,
                    apple_original_transaction_id,
                    apple_app_account_token,
                    apple_subscription_status,
                    apple_expires_at,
                    apple_environment,
                    apple_last_verified_at,
                    source,
                    created_at,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                )
                ON CONFLICT (user_id) DO UPDATE SET
                    email = EXCLUDED.email,
                    stripe_customer_id = EXCLUDED.stripe_customer_id,
                    plan_tier = EXCLUDED.plan_tier,
                    subscription_status = EXCLUDED.subscription_status,
                    price_id = EXCLUDED.price_id,
                    stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                    current_period_end = EXCLUDED.current_period_end,
                    cancel_at_period_end = EXCLUDED.cancel_at_period_end,
                    legacy_grace_expires_at = EXCLUDED.legacy_grace_expires_at,
                    apple_tier = EXCLUDED.apple_tier,
                    apple_product_id = EXCLUDED.apple_product_id,
                    apple_transaction_id = EXCLUDED.apple_transaction_id,
                    apple_original_transaction_id = EXCLUDED.apple_original_transaction_id,
                    apple_app_account_token = EXCLUDED.apple_app_account_token,
                    apple_subscription_status = EXCLUDED.apple_subscription_status,
                    apple_expires_at = EXCLUDED.apple_expires_at,
                    apple_environment = EXCLUDED.apple_environment,
                    apple_last_verified_at = EXCLUDED.apple_last_verified_at,
                    source = EXCLUDED.source,
                    updated_at = NOW()
                RETURNING user_id, email, stripe_customer_id, plan_tier, subscription_status,
                          price_id, stripe_subscription_id, current_period_end,
                          cancel_at_period_end, legacy_grace_expires_at,
                          apple_tier, apple_product_id, apple_transaction_id,
                          apple_original_transaction_id, apple_app_account_token,
                          apple_subscription_status, apple_expires_at,
                          apple_environment, apple_last_verified_at, source,
                          created_at, updated_at
                """,
                (
                    user_id,
                    merged["email"],
                    merged.get("stripe_customer_id"),
                    merged.get("plan_tier", "free"),
                    merged.get("subscription_status", "none"),
                    merged.get("price_id"),
                    merged.get("stripe_subscription_id"),
                    merged.get("current_period_end"),
                    merged.get("cancel_at_period_end", False),
                    merged.get("legacy_grace_expires_at"),
                    merged.get("apple_tier"),
                    merged.get("apple_product_id"),
                    merged.get("apple_transaction_id"),
                    merged.get("apple_original_transaction_id"),
                    merged.get("apple_app_account_token"),
                    merged.get("apple_subscription_status"),
                    merged.get("apple_expires_at"),
                    merged.get("apple_environment"),
                    merged.get("apple_last_verified_at"),
                    merged.get("source", "system"),
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError(f"Failed to upsert billing state for user_id={user_id}")
    return dict(row)


def upsert_state_by_customer_id(
    stripe_customer_id: str,
    updates: Optional[dict[str, Any]] = None,
    source: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    state = get_state_by_customer_id(stripe_customer_id)
    if not state:
        return None
    merged_updates = dict(updates or {})
    merged_updates["stripe_customer_id"] = stripe_customer_id
    return upsert_customer_state(
        user_id=state["user_id"],
        email=state["email"],
        updates=merged_updates,
        source=source,
    )


def list_states_with_expired_legacy_grace(
    now: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    if not is_enabled():
        return []

    cutoff = _as_utc(now or datetime.now(timezone.utc))
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT user_id, email, stripe_customer_id, plan_tier, subscription_status,
                       price_id, stripe_subscription_id, current_period_end,
                       cancel_at_period_end, legacy_grace_expires_at,
                       apple_tier, apple_product_id, apple_transaction_id,
                       apple_original_transaction_id, apple_app_account_token,
                       apple_subscription_status, apple_expires_at,
                       apple_environment, apple_last_verified_at, source,
                       created_at, updated_at
                FROM {BILLING_STATE_TABLE}
                WHERE legacy_grace_expires_at IS NOT NULL
                  AND legacy_grace_expires_at <= %s
                  AND subscription_status = 'legacy_grace'
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def register_webhook_event(
    stripe_event_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> bool:
    """Insert webhook event for idempotency.

    Returns True when event is newly inserted, False when already exists.
    """
    if not is_enabled():
        return False

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {WEBHOOK_EVENTS_TABLE} (
                    stripe_event_id,
                    event_type,
                    payload,
                    processed_at,
                    processing_status,
                    error_message
                ) VALUES (%s, %s, %s, NULL, 'processing', NULL)
                ON CONFLICT (stripe_event_id) DO NOTHING
                """,
                (stripe_event_id, event_type, Json(payload)),
            )
            inserted = cur.rowcount > 0
        conn.commit()
    return inserted


def mark_webhook_event(
    stripe_event_id: str,
    processing_status: str,
    error_message: Optional[str] = None,
) -> None:
    if not is_enabled():
        return

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {WEBHOOK_EVENTS_TABLE}
                SET processed_at = NOW(),
                    processing_status = %s,
                    error_message = %s
                WHERE stripe_event_id = %s
                """,
                (processing_status, error_message, stripe_event_id),
            )
        conn.commit()
