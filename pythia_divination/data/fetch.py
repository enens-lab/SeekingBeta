from typing import Optional, Sequence
import io
import logging
import time
import pandas as pd
import requests

# Primary provider
import yfinance as yf

# Providers
from data.providers import alpaca as provider_alpaca
from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
SLEEP_SEC = 1.5
HTTP_TIMEOUT = 20

STOOQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# -------- Shared helpers --------

# Timeframes that require aggregation from a base timeframe
AGGREGATED_TIMEFRAMES = {
    "4Hour": ("1Hour", 4),   # aggregate 4 hourly bars
    "2Day":  ("1Day", 2),    # aggregate 2 daily bars
    "3Day":  ("1Day", 3),    # aggregate 3 daily bars
}

# Native timeframes supported directly by providers
NATIVE_TIMEFRAMES = {"1Min", "5Min", "15Min", "30Min", "1Hour", "1Day", "1Week", "1Month"}

def _norm_timeframe(tf: Optional[str]) -> str:
    """
    Canonical timeframe for our internal router.

    Native timeframes (direct provider support):
      '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day', '1Week', '1Month'

    Aggregated timeframes (computed from base):
      '4Hour' (from 1Hour), '2Day' (from 1Day), '3Day' (from 1Day)
    """
    if not tf:
        return "1Day"
    tf = tf.strip()
    # already canonical?
    all_valid = NATIVE_TIMEFRAMES | set(AGGREGATED_TIMEFRAMES.keys())
    if tf in all_valid:
        return tf
    tl = tf.lower()
    map_ = {
        # Minutes
        "1m":    "1Min",
        "5m":    "5Min",
        "15m":   "15Min",
        "30m":   "30Min",
        # Hours
        "1h":    "1Hour",
        "60m":   "1Hour",
        "4h":    "4Hour",
        "240m":  "4Hour",
        # Days
        "1d":    "1Day",
        "1day":  "1Day",
        "2d":    "2Day",
        "2day":  "2Day",
        "3d":    "3Day",
        "3day":  "3Day",
        # Weeks
        "1w":    "1Week",
        "1wk":   "1Week",
        "1week": "1Week",
        # Months
        "1mo":     "1Month",
        "1month":  "1Month",
    }
    return map_.get(tl, "1Day")


def _aggregate_ohlcv(df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """
    Aggregate OHLCV data by combining n_bars into one.
    Uses standard OHLCV aggregation rules:
      - Open: first open
      - High: max high
      - Low: min low
      - Close: last close
      - Volume: sum of volumes
    """
    if n_bars <= 1 or df.empty:
        return df

    # Group into chunks of n_bars
    df = df.sort_index()
    n_groups = len(df) // n_bars

    if n_groups == 0:
        return df

    # Trim to complete groups only
    df_trimmed = df.iloc[-(n_groups * n_bars):]

    # Create group labels
    groups = [i // n_bars for i in range(len(df_trimmed))]

    agg_data = []
    for g in range(n_groups):
        chunk = df_trimmed.iloc[g * n_bars : (g + 1) * n_bars]
        agg_data.append({
            "Date": chunk.index[-1],  # use last timestamp as the bar's time
            "Open": chunk["Open"].iloc[0],
            "High": chunk["High"].max(),
            "Low": chunk["Low"].min(),
            "Close": chunk["Close"].iloc[-1],
            "Volume": chunk["Volume"].sum(),
        })

    result = pd.DataFrame(agg_data).set_index("Date")
    return result[["Open", "High", "Low", "Close", "Volume"]]

def _ensure_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    cols = {c.lower(): c for c in df.columns}
    needed = ['open','high','low','close','volume']
    if not all(c in cols for c in needed):
        raise ValueError(f"Expected OHLCV columns missing for {ticker}: got {list(df.columns)}")
    out = df[[cols['open'], cols['high'], cols['low'], cols['close'], cols['volume']]].copy()
    out.columns = ['Open','High','Low','Close','Volume']
    out = out[~out.index.duplicated(keep='last')]
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if getattr(out.index, 'tz', None) is not None:
        out.index = out.index.tz_convert(None)
    return out

def _yf_download(ticker: str, start: Optional[str], period: Optional[str]) -> pd.DataFrame:
    return yf.download(
        ticker,
        start=start if start else None,
        period=None if start else (period or "120d"),
        auto_adjust=True,
        progress=False,
        threads=False,
        repair=True,
        interval="1d",
    )

def _yf_history(ticker: str, start: Optional[str], period: Optional[str]) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if start:
        return t.history(start=start, interval="1d")
    return t.history(period=period or "120d", interval="1d")

# --- Stooq CSV fallback (no pandas-datareader) ---

def _stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    if "." not in t:
        t = f"{t}.us"
    return t

def _stooq_http(ticker: str, start: Optional[str]) -> pd.DataFrame:
    sym = _stooq_symbol(ticker)
    urls = [
        f"https://stooq.com/q/d/l/?s={sym}&i=d",
        f"https://stooq.com/q/l/?s={sym}&f=sd2t2ohlcv&h&e=csv",
    ]
    errors: list[str] = []

    for url in urls:
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=STOOQ_HEADERS)
            resp.raise_for_status()

            body = resp.text.strip()
            if not body:
                raise ValueError("empty response body")

            # Stooq sometimes serves HTML challenge/rate-limit pages to bots.
            sample = body.lstrip()[:256].lower()
            if sample.startswith("<!doctype html") or sample.startswith("<html"):
                raise ValueError("received HTML instead of CSV (possible bot challenge)")

            df = pd.read_csv(io.StringIO(body))
            if df.empty:
                raise ValueError("parsed CSV is empty")

            cols = {str(c).strip().lower(): str(c).strip() for c in df.columns}
            if "date" not in cols:
                raise ValueError(f"missing Date column. columns={list(df.columns)}")

            df["Date"] = pd.to_datetime(df[cols["date"]], utc=False, errors="coerce")
            df = df[df["Date"].notna()]
            if df.empty:
                raise ValueError("no valid dated rows in CSV")

            df = df.set_index("Date").sort_index()
            if start:
                df = df[df.index >= pd.to_datetime(start)]

            # Normalize OHLCV columns (case-insensitive)
            cols = {str(c).strip().lower(): str(c).strip() for c in df.columns}
            normalized = pd.DataFrame(index=df.index)
            for src, dst in [
                ("open", "Open"),
                ("high", "High"),
                ("low", "Low"),
                ("close", "Close"),
                ("volume", "Volume"),
            ]:
                if src not in cols:
                    raise ValueError(f"missing {src} column. columns={list(df.columns)}")
                normalized[dst] = pd.to_numeric(df[cols[src]], errors="coerce")

            # Require price rows to be numeric.
            normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
            if normalized.empty:
                raise ValueError("no numeric OHLC rows after normalization")

            # Volume can be missing for some instruments; default to 0.
            normalized["Volume"] = normalized["Volume"].fillna(0)
            return normalized[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as exc:
            errors.append(f"{url} -> {exc}")

    msg = " | ".join(errors[-2:]) if errors else "unknown error"
    raise RuntimeError(f"Stooq fetch failed for {ticker} ({sym}). {msg}")

# -------- Public API: fetch_ohlcv / fetch_panel --------

# Mapping from canonical timeframes to yfinance intervals
YF_INTERVAL_MAP = {
    "1Min":   "1m",
    "5Min":   "5m",
    "15Min":  "15m",
    "30Min":  "30m",
    "1Hour":  "60m",
    "1Day":   "1d",
    "1Week":  "1wk",
    "1Month": "1mo",
}

def fetch_ohlcv(
    ticker: str,
    start: Optional[str] = None,
    period: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    data_source: str = "auto",
    timeframe: Optional[str] = None,
) -> pd.DataFrame:
    src = (data_source or "auto").lower()
    tf = _norm_timeframe(timeframe)

    # Check if this is an aggregated timeframe
    needs_aggregation = tf in AGGREGATED_TIMEFRAMES
    if needs_aggregation:
        base_tf, n_bars = AGGREGATED_TIMEFRAMES[tf]
        # Fetch more data to have enough bars after aggregation
        if period:
            # Multiply the period to get enough base bars
            days = int("".join(ch for ch in period if ch.isdigit()) or "120")
            period = f"{days * n_bars}d"
    else:
        base_tf = tf
        n_bars = 1

    if src == "alpaca":
        alp = settings.alpaca
        df = provider_alpaca.fetch_ohlcv(
            ticker,
            base_url=alp.base_url,
            key_id=alp.key_id,
            secret_key=alp.secret_key,
            start=start,
            period=period,
            feed=alp.feed,
            timeframe=base_tf,
        )
        if needs_aggregation:
            df = _aggregate_ohlcv(df, n_bars)
        return df

    if src == "stooq":
        if base_tf != "1Day":
            raise RuntimeError("Stooq provider supports only daily timeframe.")
        last_err: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            try:
                d = _stooq_http(ticker, start)
                df = _ensure_ohlcv(d, ticker)
                if needs_aggregation:
                    df = _aggregate_ohlcv(df, n_bars)
                return df
            except Exception as e:
                last_err = e
                logger.warning(
                    "Stooq fetch attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    max(1, retries),
                    ticker,
                    e,
                )
                time.sleep(SLEEP_SEC)
        raise RuntimeError(f"Failed to fetch OHLCV (Stooq) for {ticker}. Last error: {last_err}")

    if src == "yahoo":
        # Map our canonical tf to yfinance interval
        if base_tf not in YF_INTERVAL_MAP:
            raise RuntimeError(f"Timeframe {base_tf} not supported by Yahoo provider")
        yf_interval = YF_INTERVAL_MAP[base_tf]
        for _ in range(max(1, retries)):
            try:
                # yfinance quirk: intraday often needs a 'period' (e.g., '30d') rather than start
                if yf_interval not in ("1d", "1wk", "1mo"):
                    d = yf.download(
                        ticker,
                        period=period or "30d",
                        interval=yf_interval,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                        repair=True,
                    )
                else:
                    d = yf.download(
                        ticker,
                        start=start if start else None,
                        period=None if start else (period or "120d"),
                        interval=yf_interval,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                        repair=True,
                    )
                df = _ensure_ohlcv(d, ticker)
                if needs_aggregation:
                    df = _aggregate_ohlcv(df, n_bars)
                return df
            except Exception:
                time.sleep(SLEEP_SEC)
        try:
            d = _yf_history(ticker, start, period)
            df = _ensure_ohlcv(d, ticker)
            if needs_aggregation:
                df = _aggregate_ohlcv(df, n_bars)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to fetch OHLCV (Yahoo) for {ticker}. Last error: {e}")

    # auto: try yahoo → stooq with chosen tf
    if base_tf not in YF_INTERVAL_MAP:
        raise RuntimeError(f"Timeframe {base_tf} not supported by auto provider")
    yf_interval = YF_INTERVAL_MAP[base_tf]
    yahoo_err: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            if yf_interval not in ("1d", "1wk", "1mo"):
                d = yf.download(
                    ticker,
                    period=period or "30d",
                    interval=yf_interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    repair=True,
                )
            else:
                d = _yf_download(ticker, start, period)
            df = _ensure_ohlcv(d, ticker)
            if needs_aggregation:
                df = _aggregate_ohlcv(df, n_bars)
            return df
        except Exception as e:
            yahoo_err = e
            time.sleep(SLEEP_SEC)
    try:
        d = _yf_history(ticker, start, period)
        df = _ensure_ohlcv(d, ticker)
        if needs_aggregation:
            df = _aggregate_ohlcv(df, n_bars)
        return df
    except Exception as e:
        yahoo_err = e
        pass
    if base_tf != "1Day":
        raise RuntimeError(
            f"Auto provider fallback to Stooq only supports daily timeframe. "
            f"Yahoo error: {yahoo_err}"
        )
    stooq_err: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            d = _stooq_http(ticker, start)
            df = _ensure_ohlcv(d, ticker)
            if needs_aggregation:
                df = _aggregate_ohlcv(df, n_bars)
            return df
        except Exception as e:
            stooq_err = e
            logger.warning(
                "Auto fallback Stooq attempt %d/%d failed for %s: %s",
                attempt + 1,
                max(1, retries),
                ticker,
                e,
            )
            time.sleep(SLEEP_SEC)
    raise RuntimeError(
        f"Failed to fetch OHLCV for {ticker} after retries and fallbacks. "
        f"Yahoo error: {yahoo_err}; Stooq error: {stooq_err}"
    )

def fetch_ohlcv_for_lstm(
    ticker: str,
    sequence_length: int = 60,
    retries: int = DEFAULT_RETRIES,
    data_source: str = "auto",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for LSTM models with extended history for proper scaling.

    LSTM models need enough data for:
    1. Feature calculation (requires ~60 days for indicators)
    2. Sequence creation (requires sequence_length days)
    3. Scaling window (Stock_Prediction_Model uses 252 days for StandardScaler fit)

    Args:
        ticker: Stock ticker symbol
        sequence_length: Number of timesteps in LSTM sequence (default: 60)
        retries: Number of retry attempts
        data_source: Data source to use ("auto", "yahoo", "alpaca", "stooq")

    Returns:
        DataFrame with OHLCV data, at least 252 days of history
    """
    # Fetch at least 400 days to ensure we have 252+ after feature engineering dropna
    period = "400d"
    return fetch_ohlcv(
        ticker=ticker,
        period=period,
        retries=retries,
        data_source=data_source,
        timeframe="1Day",
    )


def fetch_panel(
    tickers: Sequence[str],
    start: str,
    retries: int = DEFAULT_RETRIES,
    data_source: str = "auto",
    timeframe: Optional[str] = None
) -> pd.DataFrame:
    src = (data_source or "auto").lower()
    tf = _norm_timeframe(timeframe)

    if src == "alpaca":
        alp = settings.alpaca
        return provider_alpaca.fetch_panel(
            tickers,
            base_url=alp.base_url,
            key_id=alp.key_id,
            secret_key=alp.secret_key,
            start=start,
            feed=alp.get("feed", "iex"),
            timeframe=tf,  # implemented below in provider
        )

    if src == "stooq":
        frames = {}
        for t in tickers:
            ok = False
            for _ in range(max(1, retries)):
                try:
                    df = _stooq_http(t, start)
                    frames[t] = _ensure_ohlcv(df, t)
                    ok = True
                    break
                except Exception:
                    time.sleep(SLEEP_SEC)
            if not ok:
                raise RuntimeError(f"Failed Stooq fetch for {t}")
        panel = pd.concat(frames, axis=1).swaplevel(0,1,axis=1)
        panel.columns = pd.MultiIndex.from_tuples([(c[1], c[0]) for c in panel.columns], names=['Field','Ticker'])
        return panel

    if src == "yahoo":
        for _ in range(max(1, retries)):
            try:
                data = yf.download(
                    list(tickers),
                    start=start,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by='ticker',
                    repair=True,
                    interval=tf,
                )
                return data
            except Exception:
                time.sleep(SLEEP_SEC)
        raise RuntimeError(f"Failed to fetch panel (Yahoo) for {tickers}")

    # auto: try yahoo → stooq
    for _ in range(max(1, retries)):
        try:
            data = yf.download(
                list(tickers),
                start=start,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by='ticker',
                repair=True,
                interval=tf,
            )
            return data
        except Exception:
            time.sleep(SLEEP_SEC)
    # Stooq fallback
    frames = {}
    for t in tickers:
        ok = False
        for _ in range(max(1, retries)):
            try:
                df = _stooq_http(t, start)
                frames[t] = _ensure_ohlcv(df, t)
                ok = True
                break
            except Exception:
                time.sleep(SLEEP_SEC)
        if not ok:
            raise RuntimeError(f"Failed fallback fetch for {t}")
    panel = pd.concat(frames, axis=1).swaplevel(0,1,axis=1)
    panel.columns = pd.MultiIndex.from_tuples([(c[1], c[0]) for c in panel.columns], names=['Field','Ticker'])
    return panel
