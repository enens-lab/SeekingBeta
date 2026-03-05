"""Postgres-backed storage for public beta tester signups."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from .logging_config import get_logger

logger = get_logger("beta_program_store")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
BETA_TESTERS_TABLE = "beta_tester_applications"


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


def _clean_text(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    return cleaned[:limit]


def ensure_table() -> bool:
    if not is_enabled():
        logger.warning("DATABASE_URL not set; beta program store disabled")
        return False

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {BETA_TESTERS_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            role TEXT NULL,
            organization TEXT NULL,
            investing_experience TEXT NULL,
            testing_focus TEXT NOT NULL,
            source TEXT NULL,
            status TEXT NOT NULL DEFAULT 'new',
            ip_address TEXT NULL,
            user_agent TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    create_indexes_sql = [
        f"""
        CREATE INDEX IF NOT EXISTS idx_{BETA_TESTERS_TABLE}_status
        ON {BETA_TESTERS_TABLE}(status)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{BETA_TESTERS_TABLE}_created_at
        ON {BETA_TESTERS_TABLE}(created_at DESC)
        """,
    ]

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            for statement in create_indexes_sql:
                cur.execute(statement)
        conn.commit()

    logger.info("Beta tester applications table ready")
    return True


def upsert_beta_tester(
    *,
    email: str,
    full_name: str,
    role: Optional[str],
    organization: Optional[str],
    investing_experience: Optional[str],
    testing_focus: str,
    source: Optional[str],
    ip_address: Optional[str],
    user_agent: Optional[str],
) -> dict[str, Any]:
    if not is_enabled():
        raise RuntimeError("Beta program store disabled")

    safe_email = _normalize_email(email)
    safe_name = _clean_text(full_name, 120)
    if not safe_name:
        raise ValueError("full_name is required")

    safe_role = _clean_text(role, 120)
    safe_org = _clean_text(organization, 160)
    safe_experience = _clean_text(investing_experience, 80)
    safe_focus = _clean_text(testing_focus, 1200)
    if not safe_focus:
        raise ValueError("testing_focus is required")

    safe_source = _clean_text(source, 120)
    safe_ip = _clean_text(ip_address, 80)
    safe_ua = _clean_text(user_agent, 512)

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id
                FROM {BETA_TESTERS_TABLE}
                WHERE email = %s
                """,
                (safe_email,),
            )
            existing = cur.fetchone()
            created = existing is None

            if created:
                cur.execute(
                    f"""
                    INSERT INTO {BETA_TESTERS_TABLE} (
                        email,
                        full_name,
                        role,
                        organization,
                        investing_experience,
                        testing_focus,
                        source,
                        status,
                        ip_address,
                        user_agent,
                        created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, 'new', %s, %s, NOW(), NOW()
                    )
                    RETURNING id, email, full_name, status, created_at, updated_at
                    """,
                    (
                        safe_email,
                        safe_name,
                        safe_role,
                        safe_org,
                        safe_experience,
                        safe_focus,
                        safe_source,
                        safe_ip,
                        safe_ua,
                    ),
                )
            else:
                cur.execute(
                    f"""
                    UPDATE {BETA_TESTERS_TABLE}
                    SET
                        full_name = %s,
                        role = %s,
                        organization = %s,
                        investing_experience = %s,
                        testing_focus = %s,
                        source = COALESCE(%s, source),
                        ip_address = COALESCE(%s, ip_address),
                        user_agent = COALESCE(%s, user_agent),
                        updated_at = NOW()
                    WHERE email = %s
                    RETURNING id, email, full_name, status, created_at, updated_at
                    """,
                    (
                        safe_name,
                        safe_role,
                        safe_org,
                        safe_experience,
                        safe_focus,
                        safe_source,
                        safe_ip,
                        safe_ua,
                        safe_email,
                    ),
                )

            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError("Unable to persist beta tester application")

    out = dict(row)
    out["created"] = created
    return out
