"""Daily options-feature archival for lstm_quant.

Once a day, snapshot the 10 options features (the same ones lstm_quant consumes)
for the configured universe and persist them to Postgres — one row per symbol per
trade day. This rebuilds a ThetaData-like options history from the live Schwab
(real greeks) / yfinance (Black-Scholes) reconstructions, so future retraining /
backtesting has real options data again. The day's rows are then mirrored to S3
so the archive survives the box.

Each row records which `source` produced it (schwab vs yfinance) and whether the
features were actually reconstructed (`options_live`) — important so a research
consumer never conflates real-greek rows with BS-greek (or zero-filled) ones.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def snapshot_symbol(symbol: str, source: str = "schwab") -> dict | None:
    """Reconstruct today's 10 options features for one symbol.

    Mirrors the prediction path's source resolution: prefer Schwab (real greeks)
    when configured + a token is available, else fall back to yfinance. Returns
    {features: {...10...}, underlying_price, source, options_live} or None if the
    chain could not be fetched from any vendor."""
    from .options_features import FEATURE_COLUMNS, extract_from_frames
    from .options_live import fetch_chain_schwab, fetch_chain_yfinance

    used = "yfinance"
    g = oi = None
    if source in ("schwab", "auto"):
        try:
            from .schwab_chain import fetch_option_chain_json
            chain = fetch_option_chain_json(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Schwab chain fetch errored for %s: %s", symbol, exc)
            chain = None
        if chain is not None:
            g, oi = fetch_chain_schwab(symbol, chain)
            used = "schwab"
    if g is None:
        g, oi = fetch_chain_yfinance(symbol)
        used = "yfinance"
    if g is None:
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    feats = extract_from_frames(g, oi, today)
    if not feats:
        return None

    underlying = None
    try:
        underlying = float(g["underlying_price"].iloc[0])
    except Exception:
        underlying = None

    options_live = any(abs(float(feats.get(c, 0.0) or 0.0)) > 0 for c in FEATURE_COLUMNS)
    return {
        "features": feats,
        "underlying_price": underlying,
        "source": used,
        "options_live": options_live,
    }


def run_daily_archive(symbols, trade_date, source: str = "schwab") -> dict[str, Any]:
    """Snapshot + upsert every symbol for `trade_date`. Returns a summary dict."""
    from storage.market_data_store import is_enabled, upsert_options_features_daily

    summary: dict[str, Any] = {
        "trade_date": str(trade_date),
        "requested": len(symbols),
        "stored": 0,
        "schwab": 0,
        "yfinance": 0,
        "failed": 0,
        "errors": [],
    }
    if not is_enabled():
        summary["errors"].append("market data store disabled (no DATABASE_URL)")
        return summary

    for symbol in symbols:
        try:
            snap = snapshot_symbol(symbol, source=source)
            if not snap:
                summary["failed"] += 1
                continue
            upsert_options_features_daily(
                symbol=symbol,
                trade_date=trade_date,
                source=snap["source"],
                features=snap["features"],
                underlying_price=snap["underlying_price"],
                options_live=snap["options_live"],
            )
            summary["stored"] += 1
            summary[snap["source"]] = summary.get(snap["source"], 0) + 1
        except Exception as exc:  # pragma: no cover - per-symbol resilience
            summary["failed"] += 1
            if len(summary["errors"]) < 20:
                summary["errors"].append({"symbol": symbol, "error": str(exc)})
    return summary


def export_day_to_s3(trade_date) -> str | None:
    """Mirror a day's archived rows to S3 as parquet. Returns the s3 URI or None.

    Uses the awscli already baked into the image (S3_BUCKET + AWS_* env). Best
    effort — a failure here never breaks the archive run (Postgres stays primary)."""
    from storage.market_data_store import fetch_options_features_for_date

    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        logger.info("S3 mirror skipped: S3_BUCKET not set")
        return None
    df = fetch_options_features_for_date(trade_date)
    if df is None or df.empty:
        logger.info("S3 mirror skipped: no rows for %s", trade_date)
        return None

    prefix = os.getenv("OPTIONS_ARCHIVE_S3_PREFIX", "options_archive").strip("/")
    key = f"{prefix}/dt={trade_date}/options_features_{trade_date}.parquet"
    s3_uri = f"s3://{bucket}/{key}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / f"options_features_{trade_date}.parquet"
            df.to_parquet(local, index=False)
            region = os.getenv("AWS_REGION", "us-east-1")
            subprocess.run(
                ["aws", "s3", "cp", str(local), s3_uri, "--region", region],
                check=True, capture_output=True, text=True,
            )
        logger.info("Archived %d options rows to %s", len(df), s3_uri)
        return s3_uri
    except subprocess.CalledProcessError as exc:  # pragma: no cover - network/creds
        logger.warning("S3 mirror failed (%s): %s", s3_uri, (exc.stderr or "").strip()[:300])
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("S3 mirror failed (%s): %s", s3_uri, exc)
        return None
