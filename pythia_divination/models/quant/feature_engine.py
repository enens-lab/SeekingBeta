"""48-feature engine for the options-enriched quant LSTM — ported from
TradeBot's `app/services/feature_engine.py`.

Builds the model's 48 features from OHLCV: 38 base (technical / log-returns /
momentum / volume-profile / squeeze / VIX) computed in pure pandas, plus the 10
options-flow features reconstructed live (see `options_live`). External features
TradeBot sourced from FinViz/SEC/Senate (sentiment, insider, congress, num_
articles) are not collected in pythia and are zero-filled — exactly how TradeBot
itself defaults them when those sources are absent.

The 48-feature order is taken from the model's saved `feature_columns.json`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .options_live import options_window
from .options_features import FEATURE_COLUMNS as OPTION_FEATURE_COLUMNS

# Externals pythia does not collect live -> zero-filled (TradeBot defaults them
# identically when FinViz/SEC/Senate data is missing).
_ZERO_FILL_EXTERNALS = [
    "sentiment", "num_articles",
    "congress_buys_30d", "congress_sells_30d", "congress_net_signal", "congress_total_amount",
    "insider_shares", "insider_amount", "insider_buy_flag",
]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger(series: pd.Series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bandwidth = (upper - lower) / sma
    percent_b = (series - lower) / (upper - lower)
    return lower, sma, upper, bandwidth, percent_b


def _atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def engineer_quant_features(
    ohlcv_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    symbol: str | None = None,
    vix_df: pd.DataFrame | None = None,
    options_source: str = "auto",
    schwab_chain_json: dict | None = None,
) -> pd.DataFrame:
    """Transform raw OHLCV into the model's 48-feature frame (ordered to
    `feature_cols`). Options features are reconstructed live for `symbol`."""
    df = ohlcv_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    close, high, low, volume, opn = df["close"], df["high"], df["low"], df["volume"], df["open"]

    # ── VIX / fear index ──
    if vix_df is not None and not vix_df.empty:
        vix = vix_df.copy()
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        vix.set_index("date", inplace=True)
        df = df.join(vix, how="left")
        df["fear_index"] = df["fear_index"].ffill().fillna(20.0)
    else:
        df["fear_index"] = 20.0

    # ── log returns ──
    df["YesterdayCloseLogR"] = np.log(close / close.shift(1))
    df["YesterdayVolumeLogR"] = np.log(volume / volume.shift(1))

    # ── RSI / MACD / Bollinger / ATR ──
    df["rsi"] = _rsi(close, 14)
    macd_line, signal_line, histogram = _macd(close)
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = macd_line, signal_line, histogram
    bb_lower, bb_mid, bb_upper, bb_bandwidth, bb_percent = _bollinger(close)
    df["BBL_20_2.0_2.0"], df["BBM_20_2.0_2.0"], df["BBU_20_2.0_2.0"] = bb_lower, bb_mid, bb_upper
    df["BBB_20_2.0_2.0"], df["BBP_20_2.0_2.0"] = bb_bandwidth, bb_percent
    df["atr"] = _atr(high, low, close, 14)

    # ── volume / volatility / momentum ──
    vol_ma20 = volume.rolling(20).mean()
    sma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    df["bb_width"] = (((sma20 + std20 * 2) - (sma20 - std20 * 2)) / sma20).replace([np.inf, -np.inf], 0)
    df["vol_sma_20"] = vol_ma20
    df["rvol"] = (volume / (vol_ma20 + 1e-10)).fillna(1.0)
    df["high_52"] = close.rolling(window=252, min_periods=60).max()
    df["dist_to_high"] = ((close - df["high_52"]) / (df["high_52"] + 1e-10)).fillna(-1.0)
    df["roc_10"] = close.pct_change(10).fillna(0)
    for period in [3, 10, 30, 90]:
        df[f"momentum_{period}d"] = close.pct_change(period).fillna(0)
    df["momentum_accel"] = df["momentum_10d"] - df["momentum_10d"].shift(5)

    # ── volume profile (money-flow) ──
    df["mf_multiplier"] = ((close - low) - (high - close)) / (high - low + 1e-10)
    df["mf_volume"] = df["mf_multiplier"] * volume
    df["accum_dist_20"] = df["mf_volume"].rolling(20).sum()
    df["buy_sell_pressure"] = df["accum_dist_20"] / (volume.rolling(20).sum() + 1e-10)

    # ── short-squeeze score (no live short-float in pythia -> 0, as TradeBot defaults) ──
    df["squeeze_score"] = 0.0

    # ── externals pythia doesn't collect -> zero-fill ──
    for col in _ZERO_FILL_EXTERNALS:
        df[col] = 0.0

    # ── options flow: live-reconstructed (yfinance/Schwab), ffilled across window ──
    if symbol:
        opt = options_window(symbol, len(df), source=options_source, schwab_chain_json=schwab_chain_json)
        for col in OPTION_FEATURE_COLUMNS:
            df[col] = opt[col].to_numpy()
    else:
        for col in OPTION_FEATURE_COLUMNS:
            df[col] = 0.0

    # ── clean + order to the model's exact 48-feature schema ──
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0  # any unexpected column -> zero (defensive)
    return df[feature_cols]
