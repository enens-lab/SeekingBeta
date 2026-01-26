import datetime as dt
from typing import List, Dict, Any, Optional
import pandas as pd

from data.fetch import fetch_ohlcv
from storage.postgres import (
    upsert_positions_snapshot,
    insert_run_metric,
)

async def snapshot_positions(positions: List[Dict[str, Any]]) -> str:
    ts = dt.datetime.now(dt.timezone.utc)  # <-- aware datetime
    snaps = []
    for p in positions:
        qty = int(float(p.get("qty", 0)))
        avg_price = float(p.get("avg_entry_price", p.get("avg_price", 0.0)) or 0.0)
        snaps.append({"symbol": p["symbol"], "qty": qty, "avg_price": avg_price})
    await upsert_positions_snapshot(ts, snaps)
    return ts.isoformat()

async def compute_and_store_metrics(
    account: Dict[str, Any],
    positions: List[Dict[str, Any]],
    data_source: str,
    notes: Optional[str] = None,
):
    ts = dt.datetime.now(dt.timezone.utc)

    equity = None
    cash = None
    try:
        equity = float(account.get("equity")) if account.get("equity") is not None else None
    except Exception:
        pass
    try:
        cash = float(account.get("cash")) if account.get("cash") is not None else None
    except Exception:
        pass

    if equity is None:
        # best-effort equity from latest closes (sync fetch; fine to leave here because this
        # function is called via API endpoint that already runs blocking work in threadpool)
        symbols = [p["symbol"] for p in positions]
        eq = 0.0
        for s in symbols:
            try:
                df = fetch_ohlcv(s, period="30d", data_source=data_source, timeframe="1Day")
                if not df.empty:
                    eq += int(float(next((p["qty"] for p in positions if p["symbol"]==s), 0))) * float(df["Close"].iloc[-1])
            except Exception:
                continue
        equity = eq + (cash or 0.0)

    # TODO: Implement actual metric calculations:
    #   - pnl: Calculate from trade history (sum of realized gains/losses)
    #   - win_rate: wins / total_trades from trades table
    #   - turnover: trade volume / portfolio value
    #   - max_drawdown: peak-to-trough decline from equity curve
    # TODO: Add additional metrics:
    #   - win_count, loss_count: number of winning/losing trades
    #   - sharpe_ratio, sortino_ratio, calmar_ratio: risk-adjusted returns
    #   - volatility, var_95: risk metrics
    #   - avg_drawdown, drawdown_duration: extended drawdown tracking
    pnl = None
    win_rate = None
    turnover = None
    max_drawdown = None

    await insert_run_metric(
        ts_iso=ts,
        pnl=pnl,
        equity=equity,
        cash=cash,
        win_rate=win_rate,
        turnover=turnover,
        max_drawdown=max_drawdown,
        notes=notes,
    )
    return {
        "ts": ts.isoformat(),
        "equity": equity,
        "cash": cash,
        "pnl": pnl,
        "win_rate": win_rate,
        "turnover": turnover,
        "max_drawdown": max_drawdown,
        "notes": notes,
    }
