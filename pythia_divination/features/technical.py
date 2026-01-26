"""Technical feature engineering for stock price prediction."""

import numpy as np
import pandas as pd
from typing import List


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def make_features(
    df: pd.DataFrame,
    include_regression_target: bool = False,
    regression_horizon: int = 1
) -> pd.DataFrame:
    """
    Generate technical features from OHLCV data.

    Args:
        df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        include_regression_target: Whether to include regression targets
        regression_horizon: Days ahead for price/return prediction

    Returns:
        DataFrame with computed features and target columns
    """
    out = pd.DataFrame(index=df.index)

    # Momentum features
    out["ret_1d"] = df["Close"].pct_change(1)
    out["ret_5d"] = df["Close"].pct_change(5)

    # Volatility
    out["vol_10"] = df["Close"].pct_change().rolling(10).std()

    # Moving averages
    out["sma_10"] = df["Close"].rolling(10).mean()
    out["sma_20"] = df["Close"].rolling(20).mean()
    out["sma_gap"] = (out["sma_10"] - out["sma_20"]) / df["Close"]

    # RSI
    out["rsi_14"] = rsi(df["Close"], 14) / 100.0

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Volume
    out["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (
        df["Volume"].rolling(20).std() + 1e-9
    )

    # Classification target (binary: up or down)
    next_ret = df["Close"].pct_change().shift(-1)
    out["y_class"] = (next_ret > 0).astype(int)

    # Regression targets
    if include_regression_target:
        # Future return (percent change over horizon)
        out["y_return"] = df["Close"].pct_change(regression_horizon).shift(-regression_horizon)
        # Future price
        out["y_price"] = df["Close"].shift(-regression_horizon)

    # Keep 'y' as alias for y_class for backward compatibility
    out["y"] = out["y_class"]

    return out.dropna()


def get_feature_columns() -> List[str]:
    """Return the list of feature column names (excluding targets)."""
    return [
        "ret_1d",
        "ret_5d",
        "vol_10",
        "sma_10",
        "sma_20",
        "sma_gap",
        "rsi_14",
        "macd",
        "macd_sig",
        "vol_z",
    ]


def get_target_columns(include_regression: bool = False) -> List[str]:
    """Return the list of target column names."""
    targets = ["y", "y_class"]
    if include_regression:
        targets.extend(["y_return", "y_price"])
    return targets
