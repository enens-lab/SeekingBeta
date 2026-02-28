"""Track record computation for backtest/live performance reporting."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parents[1]
GENERIC_BACKTEST_CANDIDATES = [
    ROOT / "data" / "backtests" / "trade_summary_prob_strategy.csv",
    ROOT / "data" / "backtests" / "trade_summary.csv",
    ROOT / "data" / "backtests" / "backtest_results.csv",
    ROOT / "data" / "backtests" / "results.csv",
]
MODEL_BACKTEST_CANDIDATES = {
    "lstm_5d": [
        ROOT / "data" / "backtests" / "production_trade_log.csv",
        *GENERIC_BACKTEST_CANDIDATES,
    ],
    "lstm_jackpot": [
        ROOT / "data" / "backtests" / "jackpot_trade_log.csv",
        ROOT / "frontend" / "Stock_Prediction_Model" / "trade_summary_prob_strategy.csv",
        *GENERIC_BACKTEST_CANDIDATES,
    ],
}
MODEL_METRICS_CANDIDATES = {
    "lstm_5d": [ROOT / "data" / "backtests" / "production_metrics.json"],
    "lstm_jackpot": [ROOT / "data" / "backtests" / "jackpot_metrics.json"],
}
MODEL_LABELS = {
    "lstm_5d": "AI Strategy (Production LSTM 5-Day)",
    "lstm_jackpot": "AI Strategy (Jackpot LSTM 20-Day)",
}
_MODEL_ALIASES = {
    "lstm_5d": "lstm_5d",
    "production": "lstm_5d",
    "prod": "lstm_5d",
    "lstm_jackpot": "lstm_jackpot",
    "jackpot": "lstm_jackpot",
}

_CACHE: dict[str, dict[str, Any]] = {}
BACKTEST_CACHE_TTL_SECONDS = max(60, int(os.getenv("BACKTEST_CACHE_TTL_SECONDS", "86400")))


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _normalize_model(model: str | None) -> str:
    return _MODEL_ALIASES.get((model or "lstm_5d").strip().lower(), "lstm_5d")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


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


def _first_key_value(row: dict[str, Any], keys: list[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return key, row[key]
    return None, None


def _to_decimal_return(value: float | None) -> float | None:
    if value is None:
        return None
    # If absolute value is > 1, assume it's expressed in percent terms.
    if abs(value) > 1.0:
        return value / 100.0
    return value


def _normalize_trade_return(raw_key: str | None, raw_value: Any) -> float | None:
    """Normalize heterogeneous return columns into decimal form."""
    value = _get_float(raw_value)
    if value is None:
        return None
    key = (raw_key or "").strip().lower()
    # Explicit percent fields are always expressed in percent points.
    if key.endswith("_pct") or key in {"return_pct", "pnl_pct"}:
        return value / 100.0
    return _to_decimal_return(value)


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
        return _build_spy_curve_from_stooq(dates, start_value)

    start = (min(dates) - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (max(dates) + timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        frame = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False, interval="1d")
    except Exception:
        return _build_spy_curve_from_stooq(dates, start_value)

    if frame is None or frame.empty or "Close" not in frame:
        return _build_spy_curve_from_stooq(dates, start_value)

    closes = frame["Close"]
    if hasattr(closes, "columns"):
        # yfinance can return a DataFrame for single-ticker data in some versions.
        closes = closes.iloc[:, 0]
    closes = closes.dropna()
    if closes.empty:
        return _build_spy_curve_from_stooq(dates, start_value)

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
        return _build_spy_curve_from_stooq(dates, start_value)

    return [None if value is None else start_value * (value / base_close) for value in close_values]


def _build_spy_curve_from_stooq(dates: list[datetime], start_value: float) -> list[float | None]:
    if not dates or start_value <= 0:
        return [None for _ in dates]

    try:
        import requests  # type: ignore
    except Exception:
        return [None for _ in dates]

    try:
        resp = requests.get("https://stooq.com/q/d/l/?s=spy.us&i=d", timeout=6)
    except Exception:
        return [None for _ in dates]

    if resp.status_code != 200 or not resp.text:
        return [None for _ in dates]

    parsed_rows: list[tuple[datetime, float]] = []
    reader = csv.DictReader(resp.text.splitlines())
    for row in reader:
        dt = _get_datetime(row.get("Date"))
        close = _get_float(row.get("Close"))
        if dt is None or close is None or close <= 0:
            continue
        parsed_rows.append((dt, close))

    parsed_rows.sort(key=lambda item: item[0])
    if not parsed_rows:
        return [None for _ in dates]

    close_values: list[float | None] = []
    cursor = 0
    latest_close: float | None = None

    for target in dates:
        while cursor < len(parsed_rows) and parsed_rows[cursor][0] <= target:
            latest_close = parsed_rows[cursor][1]
            cursor += 1
        close_values.append(latest_close)

    base_close = next((v for v in close_values if v is not None and v > 0), None)
    if base_close is None:
        return [None for _ in dates]

    return [None if value is None else start_value * (value / base_close) for value in close_values]


def _build_trade_ledger_curve(
    rows: list[dict[str, Any]],
    *,
    start_value: float,
    round_trip_cost: float,
) -> tuple[list[datetime], list[float]] | None:
    """
    Build an equity curve from trade ledger rows (BUY size + SELL return_pct).

    This matches the common backtest trade-log format where each SELL row reports
    return for a previously opened BUY position.
    """
    if start_value <= 0:
        return None

    open_lots: dict[str, deque[float]] = defaultdict(deque)
    daily_pnl: dict[str, float] = defaultdict(float)
    default_slot_size = start_value / 10.0
    saw_action = False

    for row in rows:
        action = str(_first_value(row, ["action", "side", "order_side"]) or "").strip().upper()
        ticker = str(_first_value(row, ["ticker", "symbol"]) or "").strip().upper()
        trade_dt = _get_datetime(_first_value(row, ["date", "timestamp", "datetime", "sell_date", "exit_date", "closed_at"]))
        if action not in {"BUY", "SELL"} or not ticker or trade_dt is None:
            continue
        saw_action = True

        if action == "BUY":
            size = _get_float(_first_value(row, ["size", "position_size", "notional", "capital", "amount"]))
            if size is None or size <= 0:
                size = default_slot_size
            open_lots[ticker].append(float(size))
            continue

        # SELL
        raw_key, raw_value = _first_key_value(
            row,
            [
                "actual_return",
                "actual_return_pct",
                "strategy_return",
                "return",
                "return_pct",
                "pnl_return",
                "pnl_pct",
                "trade_return",
            ],
        )
        ret = _normalize_trade_return(raw_key, raw_value)
        if ret is None:
            continue

        returns_are_already_net = raw_key in {"return_pct", "pnl_pct"}
        net_ret = ret if returns_are_already_net else (ret - round_trip_cost)

        size = open_lots[ticker].popleft() if open_lots[ticker] else default_slot_size
        daily_pnl[trade_dt.date().isoformat()] += size * net_ret

    if not saw_action or not daily_pnl:
        return None

    equity = float(start_value)
    curve_dates: list[datetime] = []
    curve_values: list[float] = []
    for date_str in sorted(daily_pnl.keys()):
        equity += daily_pnl[date_str]
        curve_dates.append(datetime.fromisoformat(date_str))
        curve_values.append(float(equity))

    return curve_dates, curve_values


def _candidate_backtest_paths(model: str) -> list[Path]:
    model_key = _normalize_model(model)
    env_suffix = model_key.upper()
    candidates: list[Path] = []

    model_env_path = os.getenv(f"BACKTEST_RESULTS_PATH_{env_suffix}")
    if model_env_path:
        candidates.append(Path(model_env_path))

    env_path = os.getenv("BACKTEST_RESULTS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(MODEL_BACKTEST_CANDIDATES.get(model_key, GENERIC_BACKTEST_CANDIDATES))
    return _dedupe_paths(candidates)


def _resolve_backtest_path(model: str) -> Path | None:
    for candidate in _candidate_backtest_paths(model):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _candidate_metrics_paths(model: str) -> list[Path]:
    model_key = _normalize_model(model)
    env_suffix = model_key.upper()
    candidates: list[Path] = []

    model_env_path = os.getenv(f"BACKTEST_METRICS_PATH_{env_suffix}")
    if model_env_path:
        candidates.append(Path(model_env_path))

    env_path = os.getenv("BACKTEST_METRICS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(MODEL_METRICS_CANDIDATES.get(model_key, []))
    return _dedupe_paths(candidates)


def _load_metrics(model: str) -> dict[str, Any] | None:
    for candidate in _candidate_metrics_paths(model):
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            with candidate.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _compute_summary(path: Path, transaction_cost_bps: float, model: str) -> dict[str, Any]:
    rows = _read_rows(path)
    path_stat = path.stat()
    round_trip_cost = (transaction_cost_bps / 10_000.0) * 2.0
    model_key = _normalize_model(model)

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
        raw_return_key, raw_return_value = _first_key_value(
            row,
            [
                "actual_return",
                "actual_return_pct",
                "strategy_return",
                "return",
                "return_pct",
                "pnl_return",
                "pnl_pct",
                "trade_return",
            ],
        )
        raw_return = _normalize_trade_return(raw_return_key, raw_return_value)

        if log_return is not None:
            gross = math.exp(log_return) - 1.0
            returns_are_already_net = False
        else:
            gross = raw_return
            returns_are_already_net = raw_return_key in {"return_pct", "pnl_pct"}

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

        net = gross if returns_are_already_net else (gross - round_trip_cost)
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

    # Prefer explicit balances; otherwise build equity from trade ledger when possible.
    model_curve_values: list[float] = []
    if any(v is not None and v > 0 for v in reported_balances):
        for reported_balance, compounded in zip(reported_balances, net_curve):
            if reported_balance is not None and reported_balance > 0:
                model_curve_values.append(reported_balance)
            else:
                model_curve_values.append(start_value * compounded)
    else:
        ledger_curve = _build_trade_ledger_curve(rows, start_value=start_value, round_trip_cost=round_trip_cost)
        if ledger_curve is not None:
            trade_dates, model_curve_values = ledger_curve
        else:
            model_curve_values = [start_value * compounded for compounded in net_curve]

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

    if "unknown" in regime_breakdown:
        unknown_stats = regime_breakdown.pop("unknown")
        if len(regime_breakdown) == 0:
            regime_breakdown["all_trades"] = unknown_stats
        else:
            regime_breakdown["unclassified"] = unknown_stats
            notes.append("Some trades could not be mapped to a market regime and are shown as 'Unclassified'.")

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

    metrics = _load_metrics(model_key)
    if metrics:
        mapped_fields = {
            "n_trades": "sample_size",
            "win_rate": "hit_rate",
            "sharpe": "sharpe_ratio",
            "max_drawdown": "max_drawdown",
            "total_return": "total_return_net",
        }
        for source_key, target_key in mapped_fields.items():
            value = _get_float(metrics.get(source_key))
            if value is not None:
                if target_key == "sample_size":
                    summary[target_key] = int(value)
                else:
                    summary[target_key] = value

        if summary.get("total_return_net") is not None:
            summary["total_return_gross"] = summary["total_return_net"]

        notes.append("Metrics sidecar loaded for summary fields.")

        target_final_value = _get_float(metrics.get("final_value"))
        if (
            target_final_value is not None
            and target_final_value > 0
            and model_curve_values
            and len(model_curve_values) >= 2
            and abs(model_curve_values[-1] - start_value) > 1e-9
        ):
            # Affine rescale so curve starts at start_value and ends at metrics final_value.
            raw_end = model_curve_values[-1]
            factor = (target_final_value - start_value) / (raw_end - start_value)
            model_curve_values = [
                start_value + ((value - start_value) * factor)
                for value in model_curve_values
            ]

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
            "model_label": MODEL_LABELS.get(model_key, "AI Strategy"),
            "benchmark_label": "S&P 500 (SPY)",
            "start_value": start_value,
            "end_value": model_curve_values[-1] if model_curve_values else None,
            "benchmark_end_value": benchmark_end_value,
        },
        "message": None,
    }


def get_track_record(model: str = "lstm_5d") -> dict[str, Any]:
    model_key = _normalize_model(model)
    path = _resolve_backtest_path(model_key)
    if path is None:
        return {
            "available": False,
            "summary": None,
            "message": (
                "No backtest results found for requested model. "
                "Set BACKTEST_RESULTS_PATH or place model CSV files in data/backtests/."
            ),
        }

    mtime = path.stat().st_mtime
    cache_entry = _CACHE.setdefault(
        model_key,
        {"path": None, "mtime": None, "response": None, "cached_at": 0.0},
    )
    now = time.time()
    cached_path = cache_entry.get("path")
    cached_mtime = cache_entry.get("mtime")
    cached_at = float(cache_entry.get("cached_at") or 0.0)
    cache_is_fresh = (now - cached_at) <= BACKTEST_CACHE_TTL_SECONDS
    if (
        cached_path == str(path)
        and cached_mtime == mtime
        and cache_entry.get("response") is not None
        and cache_is_fresh
    ):
        return cache_entry["response"]

    transaction_cost_bps = _get_float(os.getenv("BACKTEST_TRANSACTION_COST_BPS")) or 10.0
    response = _compute_summary(path, transaction_cost_bps=transaction_cost_bps, model=model_key)
    cache_entry["path"] = str(path)
    cache_entry["mtime"] = mtime
    cache_entry["response"] = response
    cache_entry["cached_at"] = now
    return response


def get_track_record_curve(model: str = "lstm_5d") -> dict[str, Any]:
    model_key = _normalize_model(model)
    payload = get_track_record(model=model_key)
    if not payload.get("available"):
        return {
            "available": False,
            "series": [],
            "model_label": MODEL_LABELS.get(model_key, "AI Strategy"),
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
