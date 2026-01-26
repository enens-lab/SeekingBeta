"""
Async PostgreSQL storage layer using asyncpg connection pool.

All functions are async. Call from async endpoints or wrap with
`fastapi.concurrency.run_in_threadpool(asyncio.run, ...)` if needed.

Reads DATABASE_URL from `config.settings.settings.database_url`.
"""

from __future__ import annotations
import json
import datetime as dt
from typing import List, Dict, Any, Optional, Sequence, Union

import asyncpg
from config.settings import settings

DATABASE_URL = settings.database_url

_pool: Optional[asyncpg.pool.Pool] = None


async def _get_pool() -> asyncpg.pool.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=1,
            max_size=10,
        )
    return _pool


async def init_db() -> None:
    """
    Connectivity check and pool bootstrap. Migrations are handled by your runner.
    """
    global _pool
    # Reset pool to ensure it's created in the current event loop
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
        _pool = None
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1")


# ---------- Trades ----------

async def insert_trade(
    ts: Optional[str],
    symbol: str,
    side: str,
    qty: int,
    price: Optional[float],
    status: Optional[str],
    provider_order_id: Optional[str],
    reason: Optional[str],
    raw_json: Optional[Dict[str, Any]] = None,
) -> None:
    ts_dt = _as_dt(ts)
    sql = """
        INSERT INTO trades (ts, symbol, side, qty, price, status, provider_order_id, reason, raw_json)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            sql, ts_dt, symbol, side, qty, price, status, provider_order_id, reason,
            json.dumps(raw_json) if raw_json is not None else None
        )


async def list_trades(since_iso: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        if since_iso:
            rows = await conn.fetch(
                "SELECT * FROM trades WHERE ts >= CAST($1 AS TIMESTAMPTZ) "
                "ORDER BY ts DESC LIMIT $2",
                since_iso, limit
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM trades ORDER BY ts DESC LIMIT $1",
                limit
            )
    return [dict(r) for r in rows]


# ---------- Position snapshots ----------

async def upsert_positions_snapshot(ts: Union[str, dt.datetime], snapshots: Sequence[Dict[str, Any]]) -> None:
    if not snapshots:
        return
    ts_dt = _as_dt(ts)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            stmt = await conn.prepare(
                """
                INSERT INTO positions_snapshots (ts, symbol, qty, avg_price)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (ts, symbol)
                DO UPDATE SET qty = EXCLUDED.qty, avg_price = EXCLUDED.avg_price
                """
            )
            for s in snapshots:
                await stmt.fetch(
                    ts_dt, s["symbol"], int(s["qty"]), float(s.get("avg_price", 0.0))
                )


async def latest_positions_snapshot() -> List[Dict[str, Any]]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT ts FROM positions_snapshots ORDER BY ts DESC LIMIT 1")
        if not row:
            return []
        ts = row["ts"]
        rows = await conn.fetch("SELECT * FROM positions_snapshots WHERE ts = $1", ts)
    return [dict(r) for r in rows]


# ---------- Run metrics ----------

async def insert_run_metric(
    ts_iso: Union[str, dt.datetime],
    pnl: Optional[float],
    equity: Optional[float],
    cash: Optional[float],
    win_rate: Optional[float],
    turnover: Optional[float],
    max_drawdown: Optional[float],
    notes: Optional[str] = None,
) -> None:
    ts_dt = _as_dt(ts_iso)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO run_metrics (ts, pnl, equity, cash, win_rate, turnover, max_drawdown, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (ts) DO UPDATE SET
              pnl = EXCLUDED.pnl,
              equity = EXCLUDED.equity,
              cash = EXCLUDED.cash,
              win_rate = EXCLUDED.win_rate,
              turnover = EXCLUDED.turnover,
              max_drawdown = EXCLUDED.max_drawdown,
              notes = EXCLUDED.notes
            """,
            ts_dt, pnl, equity, cash, win_rate, turnover, max_drawdown, notes
        )


async def list_metrics(limit: int = 200) -> List[Dict[str, Any]]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM run_metrics ORDER BY ts DESC LIMIT $1",
            limit
        )
    return [dict(r) for r in rows]


# ---------- Users & Auth ----------

async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get a user by username."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, password_hash, tier, created_at FROM users WHERE username = $1",
            username
        )
    return dict(row) if row else None


async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get a user by ID."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, password_hash, tier, created_at FROM users WHERE id = $1",
            user_id
        )
    return dict(row) if row else None


async def create_user(username: str, password_hash: str, tier: str = "free") -> int:
    """Create a new user and return their ID."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (username, password_hash, tier)
            VALUES ($1, $2, $3)
            ON CONFLICT (username) DO UPDATE SET password_hash = $2, tier = $3
            RETURNING id
            """,
            username, password_hash, tier
        )
    return row["id"]


async def get_rate_limit(user_id: int) -> int:
    """Get current request count for user today."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT request_count FROM rate_limits WHERE user_id = $1 AND date = CURRENT_DATE",
            user_id
        )
    return row["request_count"] if row else 0


async def increment_rate_limit(user_id: int) -> int:
    """Increment and return new request count for user today."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO rate_limits (user_id, date, request_count)
            VALUES ($1, CURRENT_DATE, 1)
            ON CONFLICT (user_id, date)
            DO UPDATE SET request_count = rate_limits.request_count + 1
            RETURNING request_count
            """,
            user_id
        )
    return row["request_count"]


# ---------- Helpers ----------

def _as_dt(ts: Optional[Union[str, dt.datetime]]) -> dt.datetime:
    """Return a timezone-aware UTC datetime."""
    if ts is None:
        return dt.datetime.now(dt.timezone.utc)
    if isinstance(ts, dt.datetime):
        # make sure it's timezone-aware UTC
        if ts.tzinfo is None:
            return ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc)
    # ts is a string — parse ISO format
    try:
        d = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        # last-resort: treat as UTC naive
        d = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d