"""Track record computation for backtest/live performance reporting."""

from __future__ import annotations

import csv
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKTEST_CANDIDATES = [
    ROOT / "frontend" / "Stock_Prediction_Model" / "trade_summary_prob_strategy.csv",
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
    round_trip_cost = (transaction_cost_bps / 10_000.0) * 2.0

    gross_returns: list[float] = []
    net_returns: list[float] = []
    daily_returns: list[float] = []
    holding_days: list[float] = []
    regime_rows: dict[str, list[float]] = {}
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
        "as_of": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z",
        "sample_size": len(net_returns),
        "transaction_cost_bps": transaction_cost_bps,
        "hit_rate": hit_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": _max_drawdown(net_curve),
        "total_return_gross": gross_curve[-1] - 1.0 if gross_curve else None,
        "total_return_net": net_curve[-1] - 1.0 if net_curve else None,
        "avg_trade_return_net": (sum(net_returns) / len(net_returns)) if net_returns else None,
        "avg_holding_days": (sum(holding_days) / len(holding_days)) if holding_days else None,
        "regime_breakdown": regime_breakdown,
        "notes": notes,
    }

    return {"available": True, "summary": summary, "message": None}


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
