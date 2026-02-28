import os, time, datetime as dt
from typing import Optional, Sequence, Dict, Any
import requests
import pandas as pd

HTTP_TIMEOUT = 20
RETRY_SLEEP = 1.5
MAX_RETRIES = 3

def _env(name: str) -> str:
    val = os.getenv(name, "")
    if not val:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return val

def _to_iso(date_like: Optional[str]) -> Optional[str]:
    if not date_like:
        return None
    try:
        d = dt.datetime.fromisoformat(date_like)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.isoformat().replace("+00:00", "Z")
    except Exception:
        return date_like

def _bars_url(base: str, symbol: str) -> str:
    # Market Data v2
    return f"{base.rstrip('/')}/stocks/{symbol}/bars"


def _symbol_candidates(symbol: str) -> list[str]:
    """Return Alpaca symbol variants to handle class-share formatting."""
    base = (symbol or "").strip().upper()
    if not base:
        return []
    candidates = [base]
    if "-" in base:
        candidates.append(base.replace("-", "."))
    if "." in base:
        candidates.append(base.replace(".", "-"))
    # preserve order while removing duplicates
    deduped = []
    seen = set()
    for s in candidates:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def _headers(key_id: str, secret_key: str) -> Dict[str, str]:
    if not key_id or not secret_key:
        raise RuntimeError("Alpaca credentials missing")
    return {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
        "User-Agent": "pythia-api-mod/1.0"
    }

def _fetch_symbol_daily(
    base_url: str,
    symbol: str,
    key_id: str,
    secret_key: str,
    start: Optional[str],
    end: Optional[str],
    feed: Optional[str],
    timeframe: str,
) -> pd.DataFrame:
    params: Dict[str, Any] = {
        "timeframe": timeframe,
        "adjustment": "all",
        "sort": "asc",
        "limit": 10000,
    }
    if start:
        params["start"] = _to_iso(start)
    if end:
        params["end"] = _to_iso(end)
    if feed:
        params["feed"] = feed  # "iex" for free plan

    url = _bars_url(base_url, symbol)
    hdrs = _headers(key_id, secret_key)

    rows = []
    token = None
    retries = 0
    while True:
        if token:
            params["page_token"] = token
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=HTTP_TIMEOUT)
            # Handle rate limit and common host errors with backoff
            if resp.status_code in (429, 502, 503, 504):
                retries += 1
                if retries > MAX_RETRIES:
                    resp.raise_for_status()
                time.sleep(RETRY_SLEEP * retries)
                continue
            resp.raise_for_status()
            js = resp.json()
            if "bars" not in js:
                raise RuntimeError(f"Alpaca response missing 'bars' (status {resp.status_code}): {js}")
            items = js.get("bars") or []
            for b in items:
                rows.append({
                    "Date": pd.to_datetime(b["t"], utc=True).tz_convert(None),
                    "Open": b["o"],
                    "High": b["h"],
                    "Low":  b["l"],
                    "Close": b["c"],
                    "Volume": b["v"],
                })
            token = js.get("next_page_token")
            if not token:
                break
        except requests.RequestException as e:
            retries += 1
            if retries > MAX_RETRIES:
                raise RuntimeError(f"Alpaca request failed after retries: {e}")
            time.sleep(RETRY_SLEEP * retries)

    if not rows:
        raise RuntimeError(f"Alpaca returned no bars for {symbol} (check symbol, feed, or permissions)")
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df[["Open","High","Low","Close","Volume"]]

def fetch_ohlcv(
    symbol: str,
    *,
    base_url: str,
    key_id: str,
    secret_key: str,
    start: Optional[str] = None,
    period: Optional[str] = None,
    feed: Optional[str] = None,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    end = None
    if period and not start:
        days = int("".join(ch for ch in period if ch.isdigit()) or "120")
        start_dt = dt.date.today() - dt.timedelta(days=days)
        start = start_dt.isoformat()
    last_error = None
    for candidate in _symbol_candidates(symbol):
        try:
            return _fetch_symbol_daily(base_url, candidate, key_id, secret_key, start, end, feed, timeframe)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Alpaca fetch failed for {symbol}: {last_error}")


def fetch_panel(
    symbols: Sequence[str],
    *,
    base_url: str,
    key_id: str,
    secret_key: str,
    start: str,
    feed: Optional[str] = None,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    frames = {}
    for s in symbols:
        last_error = None
        df = None
        for candidate in _symbol_candidates(s):
            try:
                df = _fetch_symbol_daily(base_url, candidate, key_id, secret_key, start, None, feed, timeframe)
                break
            except Exception as exc:
                last_error = exc
        if df is None:
            raise RuntimeError(f"Alpaca fetch failed for {s}: {last_error}")
        frames[s] = df
    panel = pd.concat(frames, axis=1).swaplevel(0,1,axis=1)
    panel.columns = pd.MultiIndex.from_tuples([(c[1], c[0]) for c in panel.columns], names=["Field","Ticker"])
    return panel
