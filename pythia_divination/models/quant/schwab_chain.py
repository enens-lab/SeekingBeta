"""Minimal Schwab option-chain fetcher (market data only) for lstm_quant.

Loads a `schwab-py` client from a saved OAuth token file and returns the raw
`get_option_chain` JSON (the `callExpDateMap` / `putExpDateMap` / `underlyingPrice`
structure that `options_live.fetch_chain_schwab` parses). Returns `None` whenever
Schwab is unavailable — library missing, creds unset, token missing/expired, or an
API error — so callers transparently fall back to yfinance.

Auth lifecycle (schwab-py): the access token auto-refreshes (~30 min); the refresh
token expires after 7 days and needs a re-auth (see scripts/schwab_auth.py). On a
401/403 the cached client is dropped so the next call reloads the (possibly
refreshed) token file.

Env:
  SCHWAB_APP_KEY, SCHWAB_APP_SECRET   — Schwab developer app credentials
  SCHWAB_TOKEN_PATH                   — token file (default /app/artifacts/schwab_token.json)
  SCHWAB_OPTIONS_LOOKAHEAD_DAYS       — expiries to include (default 250, covers BACK_DTE)
  SCHWAB_OPTIONS_STRIKE_COUNT         — strikes around ATM; unset/0 = full chain
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}  # cached schwab-py client (lazy)

DEFAULT_TOKEN_PATH = "/app/artifacts/schwab_token.json"
# Cover the extractor's BACK_DTE window (up to ~250 days) so iv_term_slope has back months.
LOOKAHEAD_DAYS = int(os.getenv("SCHWAB_OPTIONS_LOOKAHEAD_DAYS", "250"))
STRIKE_COUNT = int(os.getenv("SCHWAB_OPTIONS_STRIKE_COUNT", "0"))  # 0 -> full chain


def token_path() -> Path:
    return Path(os.getenv("SCHWAB_TOKEN_PATH", DEFAULT_TOKEN_PATH))


def is_configured() -> bool:
    """True if creds + a token file are present (does not validate the token)."""
    return bool(os.getenv("SCHWAB_APP_KEY") and os.getenv("SCHWAB_APP_SECRET")
                and token_path().exists())


def _load_client():
    key = os.getenv("SCHWAB_APP_KEY")
    secret = os.getenv("SCHWAB_APP_SECRET")
    if not key or not secret:
        logger.info("Schwab options disabled: SCHWAB_APP_KEY/SECRET not set")
        return None
    tp = token_path()
    if not tp.exists():
        logger.info("Schwab options disabled: token not found at %s (run scripts/schwab_auth.py)", tp)
        return None
    try:
        import schwab
        return schwab.auth.client_from_token_file(
            token_path=str(tp), api_key=key, app_secret=secret,
        )
    except Exception as exc:  # pragma: no cover - import/auth env dependent
        logger.warning("Schwab client load failed: %s", exc)
        return None


def _client():
    with _LOCK:
        if "client" not in _STATE:
            _STATE["client"] = _load_client()
        return _STATE["client"]


def _invalidate():
    with _LOCK:
        _STATE.pop("client", None)


def fetch_option_chain_json(symbol: str) -> dict | None:
    """Return Schwab's get_option_chain JSON for `symbol`, or None if unavailable.

    Never raises — any failure (no token, expired refresh token, API/network error)
    returns None so the caller falls back to yfinance."""
    client = _client()
    if client is None:
        return None
    try:
        import schwab
        now = datetime.now()
        kwargs: dict[str, Any] = {
            "contract_type": schwab.client.Client.Options.ContractType.ALL,
            "from_date": now,
            "to_date": now + timedelta(days=LOOKAHEAD_DAYS),
        }
        if STRIKE_COUNT > 0:
            kwargs["strike_count"] = STRIKE_COUNT
        resp = client.get_option_chain(symbol, **kwargs)
        code = getattr(resp, "status_code", None)
        if code != 200:
            logger.warning("Schwab get_option_chain(%s) -> HTTP %s", symbol, code)
            if code in (401, 403):  # token likely expired -> force reload next call
                _invalidate()
            return None
        data = resp.json()
        # A valid chain has the strike maps; an error body / empty chain does not.
        if not data or not (data.get("callExpDateMap") or data.get("putExpDateMap")):
            logger.warning("Schwab get_option_chain(%s) returned no strikes", symbol)
            return None
        return data
    except Exception as exc:  # pragma: no cover - network/runtime
        logger.warning("Schwab get_option_chain(%s) failed: %s", symbol, exc)
        return None


def status() -> dict:
    """Lightweight status for diagnostics (no network call)."""
    tp = token_path()
    info: dict[str, Any] = {
        "creds_set": bool(os.getenv("SCHWAB_APP_KEY") and os.getenv("SCHWAB_APP_SECRET")),
        "token_path": str(tp),
        "token_exists": tp.exists(),
    }
    if tp.exists():
        try:
            mtime = datetime.fromtimestamp(tp.stat().st_mtime)
            age_days = (datetime.now() - mtime).total_seconds() / 86400.0
            info["token_age_days"] = round(age_days, 2)
            info["refresh_days_left"] = round(max(0.0, 7.0 - age_days), 2)  # refresh token ~7d
        except Exception:
            pass
    return info
