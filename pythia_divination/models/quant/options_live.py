"""Live options-feature reconstruction — ported from TradeBot's
`app/services/options_live.py`.

ThetaData (the training vendor) is gone, so at inference we reconstruct the same
10 options features from a different vendor's option chain — yfinance by default
(free; greeks computed via Black-Scholes since yfinance has no greeks), or Schwab
when an authed chain JSON is supplied (real greeks, tighter iv_skew/gex). Both
are normalized to the ThetaData column schema and fed through the SHARED
`extract_from_frames`, so feature *definitions* are identical to training.

pythia has no ThetaData archive, so unlike TradeBot there's no historical
backfill: we compute today's chain once and forward-fill it across the model's
60-step window (options state moves slowly day-to-day; the most-recent row is the
high-signal one and is vendor-accurate). Leading rows zero-fill, matching how the
model saw padded history.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd

from .options_features import FEATURE_COLUMNS, extract_from_frames

logger = logging.getLogger(__name__)

RISK_FREE = 0.04  # flat short-rate for BS greeks
YF_MAX_EXPIRIES = 20


# ── Black-Scholes greeks (for vendors without greeks, e.g. yfinance) ──

def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_delta_gamma(S, K, dte_days, iv, right):
    """Vectorized BS delta & gamma. K/dte_days/iv arrays; right array CALL/PUT."""
    T = np.maximum(dte_days, 0.5) / 365.0
    iv = np.maximum(iv, 1e-4)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    delta_call = _norm_cdf(d1)
    delta = np.where(np.char.upper(right.astype(str)) == "CALL", delta_call, delta_call - 1.0)
    gamma = _norm_pdf(d1) / (S * iv * np.sqrt(T))
    return np.nan_to_num(delta, nan=0.0), np.nan_to_num(gamma, nan=0.0)


# ── yfinance chain → ThetaData schema ──

def fetch_chain_yfinance(symbol: str, max_expiries: int = YF_MAX_EXPIRIES):
    """Return (greeks_df, oi_df) in ThetaData column schema from yfinance, with
    greeks computed via Black-Scholes. (None, None) on any failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    try:
        t = yf.Ticker(symbol)
        exps = list(t.options)[:max_expiries]
        if not exps:
            return None, None
        spot = None
        try:
            spot = t.fast_info.get("lastPrice")
        except Exception:
            spot = None
        if not spot or spot <= 0:
            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if not spot:
            return None, None

        rows = []
        for exp in exps:
            try:
                ch = t.option_chain(exp)
            except Exception:
                continue
            for side, frame in (("CALL", ch.calls), ("PUT", ch.puts)):
                if frame is None or frame.empty:
                    continue
                f = frame.copy()
                f["right"] = side
                f["expiration"] = exp
                rows.append(f[["expiration", "strike", "right", "impliedVolatility",
                               "openInterest", "volume", "bid"]])
        if not rows:
            return None, None
        df = pd.concat(rows, ignore_index=True)
        df = df.rename(columns={"impliedVolatility": "implied_vol", "openInterest": "open_interest"})
        for col in ("implied_vol", "open_interest", "volume", "bid"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["underlying_price"] = float(spot)
        df["iv_error"] = 0.0
        df["symbol"] = symbol

        today = pd.Timestamp(datetime.now().date())
        dte = (pd.to_datetime(df["expiration"]) - today).dt.days.to_numpy()
        delta, gamma = bs_delta_gamma(
            float(spot), df["strike"].to_numpy(float), dte,
            df["implied_vol"].to_numpy(float), df["right"].to_numpy(),
        )
        df["delta"] = delta
        df["gamma"] = gamma

        greeks_df = df[["symbol", "expiration", "strike", "right", "implied_vol",
                        "delta", "gamma", "volume", "bid",
                        "underlying_price", "iv_error"]].copy()
        oi_df = df[["expiration", "strike", "right", "open_interest"]].copy()
        return greeks_df, oi_df
    except Exception as exc:  # pragma: no cover - network
        logger.warning("yfinance options chain fetch failed for %s: %s", symbol, exc)
        return None, None


# ── Schwab chain → ThetaData schema (real greeks; requires authed client) ──

def fetch_chain_schwab(symbol: str, schwab_chain_json: dict):
    """Normalize a Schwab get_option_chain JSON into the ThetaData schema."""
    underlying = schwab_chain_json.get("underlyingPrice", 0) or 0
    if underlying <= 0:
        return None, None
    rows = []
    for side, key in (("CALL", "callExpDateMap"), ("PUT", "putExpDateMap")):
        for exp, strikes in (schwab_chain_json.get(key) or {}).items():
            exp_date = exp.split(":")[0]
            for _strike, contracts in strikes.items():
                for c in contracts:
                    iv = c.get("volatility")
                    iv = (iv / 100.0) if iv and iv > 2 else (iv or 0.0)  # Schwab IV is %
                    rows.append({
                        "symbol": symbol, "expiration": exp_date,
                        "strike": c.get("strikePrice"), "right": side,
                        "implied_vol": iv, "delta": c.get("delta") or 0.0,
                        "gamma": c.get("gamma") or 0.0,
                        "open_interest": c.get("openInterest") or 0.0,
                        "volume": c.get("totalVolume") or 0.0,
                        "bid": c.get("bid") or 0.0,
                        "underlying_price": underlying, "iv_error": 0.0,
                    })
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    for col in ("delta", "gamma", "implied_vol"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df.loc[df[col] < -10, col] = 0.0  # clamp Schwab -999 sentinels
    oi_df = df[["expiration", "strike", "right", "open_interest"]].copy()
    greeks_df = df.drop(columns=["open_interest"])
    return greeks_df, oi_df


def compute_live_options_features(symbol: str, source: str = "auto",
                                  schwab_chain_json: dict | None = None) -> dict | None:
    """Today's 10 options features for a symbol (same schema as training) or None.
    source: "schwab" | "yfinance" | "auto" (schwab if json supplied else yfinance)."""
    if source in ("schwab", "auto") and schwab_chain_json is not None:
        g, oi = fetch_chain_schwab(symbol, schwab_chain_json)
    else:
        g, oi = fetch_chain_yfinance(symbol)
    if g is None:
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    return extract_from_frames(g, oi, today)


def options_window(symbol: str, n_rows: int, source: str = "auto",
                   schwab_chain_json: dict | None = None) -> pd.DataFrame:
    """Return an n_rows-long DataFrame of the 10 options columns for the model
    window. With no ThetaData archive, today's reconstructed values are
    forward-filled across the window; if reconstruction fails, all zeros (the
    model still runs, just without the options lift)."""
    feats = None
    try:
        feats = compute_live_options_features(symbol, source=source, schwab_chain_json=schwab_chain_json)
    except Exception as exc:  # pragma: no cover
        logger.warning("options reconstruction failed for %s: %s", symbol, exc)
    out = pd.DataFrame(0.0, index=range(n_rows), columns=list(FEATURE_COLUMNS))
    if feats:
        for col in FEATURE_COLUMNS:
            out[col] = float(feats.get(col, 0.0))
    return out
