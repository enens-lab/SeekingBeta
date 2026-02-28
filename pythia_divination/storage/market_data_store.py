"""Postgres-backed market OHLCV storage and LSTM prediction cache."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
MARKET_DATA_TABLE = os.getenv("MARKET_DATA_TABLE", "market_ohlcv_daily")
LSTM_PREDICTION_TABLE = os.getenv("LSTM_PREDICTION_TABLE", "lstm_prediction_cache")


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


def ensure_tables() -> bool:
    if not is_enabled():
        logger.warning("DATABASE_URL not set; market data store disabled")
        return False

    sql_market_table = f"""
        CREATE TABLE IF NOT EXISTS {MARKET_DATA_TABLE} (
            ticker TEXT NOT NULL,
            trade_date DATE NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (ticker, trade_date)
        )
    """
    sql_market_index = f"""
        CREATE INDEX IF NOT EXISTS idx_{MARKET_DATA_TABLE}_date
        ON {MARKET_DATA_TABLE}(trade_date DESC)
    """

    sql_prediction_table = f"""
        CREATE TABLE IF NOT EXISTS {LSTM_PREDICTION_TABLE} (
            data_source TEXT NOT NULL,
            model_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            probability DOUBLE PRECISION NULL,
            signal TEXT NULL,
            last_close DOUBLE PRECISION NULL,
            generated_at TIMESTAMPTZ NULL,
            payload JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (data_source, model_name, ticker)
        )
    """
    sql_prediction_index = f"""
        CREATE INDEX IF NOT EXISTS idx_{LSTM_PREDICTION_TABLE}_updated
        ON {LSTM_PREDICTION_TABLE}(updated_at DESC)
    """

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_market_table)
            cur.execute(sql_market_index)
            cur.execute(sql_prediction_table)
            cur.execute(sql_prediction_index)
        conn.commit()
    logger.info(
        "Market data tables ready: %s, %s",
        MARKET_DATA_TABLE,
        LSTM_PREDICTION_TABLE,
    )
    return True


def upsert_ohlcv_dataframe(ticker: str, frame: pd.DataFrame, source: str) -> int:
    """Upsert daily OHLCV rows for a ticker."""
    if not is_enabled() or frame is None or frame.empty:
        return 0

    normalized_ticker = ticker.upper()
    rows = []
    for ts, row in frame.iterrows():
        dt_value = ts.date() if isinstance(ts, datetime) else pd.Timestamp(ts).date()
        rows.append(
            (
                normalized_ticker,
                dt_value,
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row.get("Volume", 0.0) or 0.0),
                source,
            )
        )
    if not rows:
        return 0

    sql = f"""
        INSERT INTO {MARKET_DATA_TABLE} (
            ticker, trade_date, open, high, low, close, volume, source, fetched_at, updated_at
        )
        VALUES %s
        ON CONFLICT (ticker, trade_date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            source = EXCLUDED.source,
            fetched_at = NOW(),
            updated_at = NOW()
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=500)
        conn.commit()
    return len(rows)


def get_ohlcv_dataframe(ticker: str, max_rows: int = 500) -> pd.DataFrame:
    """Read latest daily OHLCV rows for ticker from Postgres."""
    if not is_enabled():
        return pd.DataFrame()

    sql = f"""
        SELECT trade_date, open, high, low, close, volume
        FROM {MARKET_DATA_TABLE}
        WHERE ticker = %s
        ORDER BY trade_date DESC
        LIMIT %s
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker.upper(), int(max_rows)))
            records = [dict(r) for r in cur.fetchall()]
    if not records:
        return pd.DataFrame()

    records.reverse()
    df = pd.DataFrame.from_records(records)
    df["Date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[df["Date"].notna()].set_index("Date")
    out = pd.DataFrame(index=df.index)
    out["Open"] = pd.to_numeric(df["open"], errors="coerce")
    out["High"] = pd.to_numeric(df["high"], errors="coerce")
    out["Low"] = pd.to_numeric(df["low"], errors="coerce")
    out["Close"] = pd.to_numeric(df["close"], errors="coerce")
    out["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out[["Open", "High", "Low", "Close", "Volume"]]


def get_latest_trade_dates(tickers: Sequence[str]) -> Dict[str, date]:
    if not is_enabled() or not tickers:
        return {}

    sql = f"""
        SELECT ticker, MAX(trade_date) AS latest_date
        FROM {MARKET_DATA_TABLE}
        WHERE ticker = ANY(%s)
        GROUP BY ticker
    """
    normalized = [t.upper() for t in tickers]
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (normalized,))
            rows = [dict(r) for r in cur.fetchall()]
    return {str(r["ticker"]).upper(): r["latest_date"] for r in rows if r.get("latest_date")}


def select_stale_tickers(
    tickers: Sequence[str],
    max_age_days: int = 1,
    limit: int = 250,
) -> list[str]:
    """Return ordered stale/missing tickers limited by batch size."""
    if not tickers:
        return []
    latest = get_latest_trade_dates(tickers)
    cutoff = date.today().toordinal() - max(1, int(max_age_days))
    stale: list[str] = []
    for raw in tickers:
        ticker = raw.upper()
        last_dt = latest.get(ticker)
        if last_dt is None:
            stale.append(ticker)
        elif last_dt.toordinal() < cutoff:
            stale.append(ticker)
        if len(stale) >= limit:
            break
    return stale


def upsert_lstm_prediction(
    *,
    data_source: str,
    model_name: str,
    ticker: str,
    payload: Dict[str, Any],
) -> None:
    if not is_enabled():
        return

    prob = payload.get("probability")
    signal = payload.get("signal")
    last_close = payload.get("last_close")
    generated_at = payload.get("generated_at")
    parsed_generated_at: Optional[datetime] = None
    if isinstance(generated_at, str):
        try:
            parsed_generated_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except Exception:
            parsed_generated_at = None
    elif isinstance(generated_at, datetime):
        parsed_generated_at = generated_at
    if parsed_generated_at and parsed_generated_at.tzinfo is None:
        parsed_generated_at = parsed_generated_at.replace(tzinfo=timezone.utc)

    sql = f"""
        INSERT INTO {LSTM_PREDICTION_TABLE} (
            data_source, model_name, ticker, probability, signal, last_close, generated_at, payload, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (data_source, model_name, ticker) DO UPDATE SET
            probability = EXCLUDED.probability,
            signal = EXCLUDED.signal,
            last_close = EXCLUDED.last_close,
            generated_at = EXCLUDED.generated_at,
            payload = EXCLUDED.payload,
            updated_at = NOW()
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    data_source.lower(),
                    model_name,
                    ticker.upper(),
                    float(prob) if prob is not None else None,
                    signal,
                    float(last_close) if last_close is not None else None,
                    parsed_generated_at,
                    Json(payload),
                ),
            )
        conn.commit()


def _coerce_payload(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            import json

            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def get_lstm_prediction(
    *,
    data_source: str,
    model_name: str,
    ticker: str,
    max_age_seconds: int,
) -> Optional[Dict[str, Any]]:
    if not is_enabled():
        return None

    sql = f"""
        SELECT payload
        FROM {LSTM_PREDICTION_TABLE}
        WHERE data_source = %s
          AND model_name = %s
          AND ticker = %s
          AND updated_at >= NOW() - (%s * INTERVAL '1 second')
        LIMIT 1
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql,
                (data_source.lower(), model_name, ticker.upper(), max(1, int(max_age_seconds))),
            )
            row = cur.fetchone()
    if not row:
        return None
    return _coerce_payload(row.get("payload"))


def list_lstm_predictions(
    *,
    data_source: str,
    model_names: Sequence[str],
    tickers: Sequence[str],
    max_age_seconds: int,
) -> Dict[tuple[str, str], Dict[str, Any]]:
    if not is_enabled() or not model_names or not tickers:
        return {}

    sql = f"""
        SELECT model_name, ticker, payload
        FROM {LSTM_PREDICTION_TABLE}
        WHERE data_source = %s
          AND model_name = ANY(%s)
          AND ticker = ANY(%s)
          AND updated_at >= NOW() - (%s * INTERVAL '1 second')
    """
    models = list(model_names)
    symbols = [t.upper() for t in tickers]
    out: Dict[tuple[str, str], Dict[str, Any]] = {}
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (data_source.lower(), models, symbols, max(1, int(max_age_seconds))))
            rows = [dict(r) for r in cur.fetchall()]
    for row in rows:
        payload = _coerce_payload(row.get("payload"))
        if not payload:
            continue
        key = (str(row["model_name"]), str(row["ticker"]).upper())
        out[key] = payload
    return out
