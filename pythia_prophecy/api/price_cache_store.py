"""Postgres-backed last-close cache for dashboard prediction endpoints."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from .logging_config import get_logger

logger = get_logger("price_cache")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
PRICE_CACHE_TABLE = os.getenv("PRICE_CACHE_TABLE", "dashboard_price_cache")


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


def ensure_table() -> bool:
    if not is_enabled():
        logger.warning("DATABASE_URL not set; Postgres price cache disabled")
        return False

    sql_table = f"""
        CREATE TABLE IF NOT EXISTS {PRICE_CACHE_TABLE} (
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'unknown',
            last_close DOUBLE PRECISION NOT NULL,
            close_date DATE NULL,
            fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (ticker, timeframe)
        )
    """
    sql_index = f"""
        CREATE INDEX IF NOT EXISTS idx_{PRICE_CACHE_TABLE}_updated_at
        ON {PRICE_CACHE_TABLE}(updated_at)
    """

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_table)
            cur.execute(sql_index)
        conn.commit()
    logger.info("Postgres price cache table ready: %s", PRICE_CACHE_TABLE)
    return True


def upsert_last_close(
    ticker: str,
    timeframe: str,
    last_close: float,
    source: str = "unknown",
    close_date: Optional[str] = None,
) -> None:
    if not is_enabled():
        return

    sql = f"""
        INSERT INTO {PRICE_CACHE_TABLE} (
            ticker, timeframe, source, last_close, close_date, fetched_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s::date, NOW(), NOW())
        ON CONFLICT (ticker, timeframe) DO UPDATE SET
            source = EXCLUDED.source,
            last_close = EXCLUDED.last_close,
            close_date = EXCLUDED.close_date,
            fetched_at = NOW(),
            updated_at = NOW()
    """

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    ticker.upper(),
                    timeframe.lower(),
                    source,
                    float(last_close),
                    close_date,
                ),
            )
        conn.commit()


def get_last_close(ticker: str, timeframe: str, max_age_hours: int = 72) -> Optional[dict]:
    if not is_enabled():
        return None

    sql = f"""
        SELECT ticker, timeframe, source, last_close, close_date, fetched_at, updated_at
        FROM {PRICE_CACHE_TABLE}
        WHERE ticker = %s
          AND timeframe = %s
          AND updated_at >= NOW() - (%s * INTERVAL '1 hour')
        LIMIT 1
    """

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker.upper(), timeframe.lower(), max_age_hours))
            row = cur.fetchone()
            return dict(row) if row else None
