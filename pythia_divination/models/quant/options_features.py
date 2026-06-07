"""Canonical 10 options-flow features — ported verbatim from TradeBot's
`lstm/featuresPy/theta_options.py` (`extract_from_frames`).

This is the SHARED definition used at both training time (against ThetaData EOD
greek chains) and live inference (against a yfinance/Schwab chain normalized to
the same column schema). Keeping the computation identical guarantees the live
features match the training distribution; only the upstream data vendor differs.

The 10 features (in model order): atm_iv, iv_skew_25d, iv_term_slope,
pc_oi_ratio, pc_volume_ratio, opt_volume_ratio, gex, net_delta_oi, total_oi,
n_contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "atm_iv",
    "iv_skew_25d",
    "iv_term_slope",
    "pc_oi_ratio",
    "pc_volume_ratio",
    "opt_volume_ratio",
    "gex",
    "net_delta_oi",
    "total_oi",
    "n_contracts",
]

FRONT_DTE_MIN, FRONT_DTE_MAX = 7, 45
BACK_DTE_MIN, BACK_DTE_MAX = 45, 250


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def extract_from_frames(g, oi, date_str: str) -> dict | None:
    """Compute the 10 options features from in-memory greeks + open-interest
    frames. `g` must carry the ThetaData greeks columns (implied_vol, delta,
    gamma, underlying_price, expiration, strike, right, bid, volume); `oi` may
    be None. Returns a dict (with a `date` key) or None when data is too thin."""
    if g is None or "implied_vol" not in g.columns:
        return None
    g = g.copy()

    # Keep only validly-priced, liquid contracts
    g = g[(g["implied_vol"] > 0.01) & (g["implied_vol"] < 5.0)]
    if "iv_error" in g.columns:
        g = g[g["iv_error"].abs() < 0.5]
    g = g[(g["bid"] > 0) | (g["volume"] > 0)]  # drop totally dead contracts
    if len(g) < 4:
        return None

    # Underlying price (most common non-null value that day)
    S = pd.to_numeric(g["underlying_price"], errors="coerce")
    S = S[S > 0]
    if S.empty:
        return None
    S = float(S.median())

    # Days to expiry
    exp = pd.to_datetime(g["expiration"], errors="coerce")
    asof = pd.to_datetime(date_str)
    g = g.assign(dte=(exp - asof).dt.days)
    g = g[g["dte"] >= 0]
    if len(g) < 4:
        return None

    # ── Open interest merge ──
    if oi is not None and "open_interest" in oi.columns:
        oi_key = oi.groupby(["expiration", "strike", "right"], as_index=False)["open_interest"].last()
        g = g.merge(oi_key, on=["expiration", "strike", "right"], how="left")
    if "open_interest" not in g.columns:
        g["open_interest"] = 0.0
    g["open_interest"] = pd.to_numeric(g["open_interest"], errors="coerce").fillna(0.0)
    calls = g[g["right"].str.upper() == "CALL"]
    puts = g[g["right"].str.upper() == "PUT"]

    # ── ATM IV (front expiry) ──
    def atm_iv(frame: pd.DataFrame, dte_lo: int, dte_hi: int) -> float | None:
        win = frame[(frame["dte"] >= dte_lo) & (frame["dte"] <= dte_hi)]
        if win.empty:
            return None
        front_exp = win.loc[(win["dte"]).idxmin(), "expiration"]
        win = win[win["expiration"] == front_exp]
        win = win.assign(moneyness=(win["strike"] - S).abs())
        win = win.nsmallest(6, "moneyness")
        return float(win["implied_vol"].mean()) if len(win) else None

    atm_front = atm_iv(g, FRONT_DTE_MIN, FRONT_DTE_MAX)
    atm_back = atm_iv(g, BACK_DTE_MIN, BACK_DTE_MAX)
    if atm_front is None:
        atm_front = float(g["implied_vol"].median())
    iv_term_slope = (atm_back - atm_front) if atm_back is not None else 0.0

    # ── 25-delta skew (front expiry) ──
    def delta_iv(frame: pd.DataFrame, target_abs_delta: float = 0.25) -> float | None:
        win = frame[(frame["dte"] >= FRONT_DTE_MIN) & (frame["dte"] <= FRONT_DTE_MAX)]
        if win.empty or "delta" not in win.columns:
            return None
        win = win.assign(dd=(win["delta"].abs() - target_abs_delta).abs())
        win = win.nsmallest(3, "dd")
        return float(win["implied_vol"].mean()) if len(win) else None

    put_25 = delta_iv(puts)
    call_25 = delta_iv(calls)
    iv_skew_25d = (put_25 - call_25) if (put_25 is not None and call_25 is not None) else 0.0

    # ── Put/Call ratios ──
    call_oi, put_oi = calls["open_interest"].sum(), puts["open_interest"].sum()
    call_vol = pd.to_numeric(calls["volume"], errors="coerce").fillna(0).sum()
    put_vol = pd.to_numeric(puts["volume"], errors="coerce").fillna(0).sum()
    total_vol = call_vol + put_vol
    total_oi = call_oi + put_oi

    # ── Greek exposures ──
    # GEX: calls add +gamma, puts add -gamma (standard dealer-positioning sign)
    sign = np.where(g["right"].str.upper() == "CALL", 1.0, -1.0)
    gamma = pd.to_numeric(g.get("gamma", 0.0), errors="coerce").fillna(0.0).to_numpy()
    delta = pd.to_numeric(g.get("delta", 0.0), errors="coerce").fillna(0.0).to_numpy()
    oi_arr = g["open_interest"].to_numpy()
    gex = float(np.sum(sign * gamma * oi_arr * 100.0 * (S ** 2) * 0.01))
    net_delta_oi = float(np.sum(delta * oi_arr * 100.0))

    return {
        "date": date_str,
        "atm_iv": round(atm_front, 6),
        "iv_skew_25d": round(iv_skew_25d, 6),
        "iv_term_slope": round(iv_term_slope, 6),
        "pc_oi_ratio": round(_safe_ratio(put_oi, call_oi), 6),
        "pc_volume_ratio": round(_safe_ratio(put_vol, call_vol), 6),
        "opt_volume_ratio": round(float(np.log1p(total_vol)), 6),
        "gex": round(gex / 1e6, 6),  # scale to $millions/1%
        "net_delta_oi": round(net_delta_oi / 1e6, 6),  # millions of shares
        "total_oi": round(float(np.log1p(total_oi)), 6),
        "n_contracts": int(len(g)),
    }
