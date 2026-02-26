"""Track record computation for backtest/live performance reporting."""

from __future__ import annotations

import csv
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKTEST_CANDIDATES = [
    ROOT / "frontend" / "Stock_Prediction_Model" / "trade_summary_prob_strategy.csv",
    ROOT / "data" / "backtests" / "jackpot_trade_log.csv",
    ROOT / "data" / "backtests" / "trade_summary_prob_strategy.csv",
    ROOT / "data" / "backtests" / "trade_summary.csv",
    ROOT / "data" / "backtests" / "backtest_results.csv",
    ROOT / "data" / "backtests" / "results.csv",
]

_CACHE: dict[str, Any] = {"path": None, "mtime": None, "response": None}


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _get_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    text = text.replace(",", "")
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _get_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {_normalize_key(k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            rows.append(normalized)
        return rows


def _first_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _to_decimal_return(value: float | None) -> float | None:
    if value is None:
        return None
    # If absolute value is > 1, assume it's expressed in percent terms.
    if abs(value) > 1.0:
        return value / 100.0
    return value


def _to_regime(row: dict[str, Any]) -> str:
    explicit = _first_value(row, ["regime", "market_regime"])
    if explicit:
        return str(explicit).strip().lower()

    market_ret = _get_float(
        _first_value(
            row,
            ["market_return", "market_return_pct", "spy_return", "spy_return_pct", "benchmark_return"],
        )
    )
    market_ret = _to_decimal_return(market_ret)
    if market_ret is None:
        return "unknown"
    if market_ret > 0.002:
        return "bull"
    if market_ret < -0.002:
        return "bear"
    return "sideways"


def _max_drawdown(curve: list[float]) -> float | None:
    if not curve:
        return None
    peak = curve[0]
    drawdown = 0.0
    for value in curve:
        if value > peak:
            peak = value
        if peak <= 0:
            continue
        current = (value / peak) - 1.0
        if current < drawdown:
            drawdown = current
    return drawdown


def _build_spy_curve(dates: list[datetime], start_value: float) -> list[float | None]:
    if not dates or start_value <= 0:
        return [None for _ in dates]

    try:
        import yfinance as yf  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return [None for _ in dates]

    start = (min(dates) - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (max(dates) + timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        frame = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False, interval="1d")
    except Exception:
        return [None for _ in dates]

    if frame is None or frame.empty or "Close" not in frame:
        return [None for _ in dates]

    closes = frame["Close"]
    if hasattr(closes, "columns"):
        # yfinance can return a DataFrame for single-ticker data in some versions.
        closes = closes.iloc[:, 0]
    closes = closes.dropna()
    if closes.empty:
        return [None for _ in dates]

    close_values: list[float | None] = []
    for dt in dates:
        timestamp = pd.Timestamp(dt)
        idx = closes.index.searchsorted(timestamp, side="right") - 1
        if idx < 0:
            close_values.append(None)
        else:
            close_values.append(float(closes.iloc[idx]))

    base_close = next((v for v in close_values if v is not None and v > 0), None)
    if base_close is None:
        return [None for _ in dates]

    return [None if value is None else start_value * (value / base_close) for value in close_values]


def _candidate_backtest_paths() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("BACKTEST_RESULTS_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(DEFAULT_BACKTEST_CANDIDATES)
    return candidates


def _resolve_backtest_path() -> Path | None:
    for candidate in _candidate_backtest_paths():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _compute_summary(path: Path, transaction_cost_bps: float) -> dict[str, Any]:
    rows = _read_rows(path)
    path_stat = path.stat()
    round_trip_cost = (transaction_cost_bps / 10_000.0) * 2.0

    gross_returns: list[float] = []
    net_returns: list[float] = []
    daily_returns: list[float] = []
    holding_days: list[float] = []
    regime_rows: dict[str, list[float]] = {}
    trade_dates: list[datetime] = []
    reported_balances: list[float | None] = []
    notes: list[str] = []

    for row in rows:
        log_return = _get_float(_first_value(row, ["actual_logr", "actual_log_return", "log_return"]))
        raw_return = _get_float(
            _first_value(
                row,
                ["actual_return", "actual_return_pct", "strategy_return", "return", "pnl_return"],
            )
        )

        if log_return is not None:
            gross = math.exp(log_return) - 1.0
        else:
            gross = _to_decimal_return(raw_return)

        if gross is None:
            continue

        trade_dt = _get_datetime(
            _first_value(
                row,
                ["date", "timestamp", "datetime", "sell_date", "exit_date", "closed_at"],
            )
        )
        if trade_dt is None:
            trade_dt = datetime.utcfromtimestamp(path_stat.st_mtime) + timedelta(minutes=len(gross_returns))
        trade_dates.append(trade_dt)

        reported_balances.append(
            _get_float(
                _first_value(
                    row,
                    ["balance", "equity", "portfolio_value", "account_value", "net_liquidation", "ending_balance"],
                )
            )
        )

        days = _get_float(_first_value(row, ["days_held", "daysheld", "holding_days", "days"]))
        if days is None or days <= 0:
            days = 1.0
        holding_days.append(days)

        net = gross - round_trip_cost
        gross_returns.append(gross)
        net_returns.append(net)

        if 1.0 + net > 0:
            daily_returns.append((1.0 + net) ** (1.0 / days) - 1.0)

        regime = _to_regime(row)
        regime_rows.setdefault(regime, []).append(net)

    if not gross_returns:
        return {
            "available": False,
            "summary": None,
            "message": f"Backtest file found at {path}, but no usable return rows were detected.",
        }

    gross_equity = 1.0
    net_equity = 1.0
    gross_curve: list[float] = []
    net_curve: list[float] = []
    for gross, net in zip(gross_returns, net_returns):
        gross_equity *= 1.0 + gross
        net_equity *= 1.0 + net
        gross_curve.append(gross_equity)
        net_curve.append(net_equity)

    start_value = _get_float(os.getenv("BACKTEST_START_CAPITAL")) or 10_000.0
    first_reported_balance = next((v for v in reported_balances if v is not None and v > 0), None)
    if first_reported_balance is not None:
        start_value = first_reported_balance

    model_curve_values: list[float] = []
    for reported_balance, compounded in zip(reported_balances, net_curve):
        if reported_balance is not None and reported_balance > 0:
            model_curve_values.append(reported_balance)
        else:
            model_curve_values.append(start_value * compounded)

    benchmark_curve_values = _build_spy_curve(trade_dates, start_value=start_value)
    benchmark_end_value = next((value for value in reversed(benchmark_curve_values) if value is not None), None)
    benchmark_total_return = (
        (benchmark_end_value / start_value - 1.0) if benchmark_end_value is not None and start_value > 0 else None
    )

    wins = sum(1 for r in net_returns if r > 0)
    losses = len(net_returns) - wins
    hit_rate = wins / len(net_returns) if net_returns else None

    sharpe_ratio: float | None = None
    if len(daily_returns) >= 2:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe_ratio = (mean / std) * math.sqrt(252.0)

    regime_breakdown: dict[str, dict[str, Any]] = {}
    for regime, values in regime_rows.items():
        regime_wins = sum(1 for v in values if v > 0)
        regime_losses = len(values) - regime_wins
        regime_breakdown[regime] = {
            "trades": len(values),
            "wins": regime_wins,
            "losses": regime_losses,
            "win_rate": (regime_wins / len(values)) if values else None,
            "avg_return_net": (sum(values) / len(values)) if values else None,
        }

    if "unknown" in regime_breakdown and len(regime_breakdown) == 1:
        notes.append("No explicit market regime columns detected; all trades grouped under 'unknown'.")

    summary = {
        "source_file": str(path),
        "as_of": datetime.utcfromtimestamp(path_stat.st_mtime).isoformat() + "Z",
        "sample_size": len(net_returns),
        "transaction_cost_bps": transaction_cost_bps,
        "hit_rate": hit_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": _max_drawdown(net_curve),
        "total_return_gross": gross_curve[-1] - 1.0 if gross_curve else None,
        "total_return_net": net_curve[-1] - 1.0 if net_curve else None,
        "benchmark_return": benchmark_total_return,
        "avg_trade_return_net": (sum(net_returns) / len(net_returns)) if net_returns else None,
        "avg_holding_days": (sum(holding_days) / len(holding_days)) if holding_days else None,
        "regime_breakdown": regime_breakdown,
        "notes": notes,
    }

    curve_points: list[dict[str, Any]] = []
    for dt, model_value, benchmark_value in zip(trade_dates, model_curve_values, benchmark_curve_values):
        curve_points.append(
            {
                "date": dt.date().isoformat(),
                "model_value": round(float(model_value), 4),
                "benchmark_value": None if benchmark_value is None else round(float(benchmark_value), 4),
            }
        )

    return {
        "available": True,
        "summary": summary,
        "curve": {
            "series": curve_points,
            "model_label": "AI Strategy (Jackpot)",
            "benchmark_label": "S&P 500 (SPY)",
            "start_value": start_value,
            "end_value": model_curve_values[-1] if model_curve_values else None,
            "benchmark_end_value": benchmark_end_value,
        },
        "message": None,
    }


def get_track_record() -> dict[str, Any]:
    path = _resolve_backtest_path()
    if path is None:
        return {
            "available": False,
            "summary": None,
            "message": "No backtest results found. Set BACKTEST_RESULTS_PATH or place a CSV in data/backtests/.",
        }

    mtime = path.stat().st_mtime
    cached_path = _CACHE.get("path")
    cached_mtime = _CACHE.get("mtime")
    if cached_path == str(path) and cached_mtime == mtime and _CACHE.get("response") is not None:
        return _CACHE["response"]

    transaction_cost_bps = _get_float(os.getenv("BACKTEST_TRANSACTION_COST_BPS")) or 10.0
    response = _compute_summary(path, transaction_cost_bps=transaction_cost_bps)
    _CACHE["path"] = str(path)
    _CACHE["mtime"] = mtime
    _CACHE["response"] = response
    return response


def get_track_record_curve() -> dict[str, Any]:
    payload = get_track_record()
    if not payload.get("available"):
        return {
            "available": False,
            "series": [],
            "model_label": "AI Strategy (Jackpot)",
            "benchmark_label": "S&P 500 (SPY)",
            "start_value": None,
            "end_value": None,
            "benchmark_end_value": None,
            "message": payload.get("message"),
        }

    curve = payload.get("curve") or {}
    series = curve.get("series") or []

    return {
        "available": len(series) > 0,
        "series": series,
        "model_label": curve.get("model_label", "AI Strategy (Jackpot)"),
        "benchmark_label": curve.get("benchmark_label", "S&P 500 (SPY)"),
        "start_value": curve.get("start_value"),
        "end_value": curve.get("end_value"),
        "benchmark_end_value": curve.get("benchmark_end_value"),
        "message": None if len(series) > 0 else "Backtest file parsed, but no chartable points were found.",
    }
