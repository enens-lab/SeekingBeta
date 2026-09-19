"""Checks for api/stock_predictions_store.py (the precomputed /predict/* source).

Plain script like the other tests here:  python tests/test_stock_predictions_store.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import stock_predictions_store as sps  # noqa: E402

CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def _pred(ticker, model, prob, ts):
    return {
        "ticker": ticker, "model": model, "description": "d", "horizon": "5 days", "target_return": ">2%",
        "probability": prob, "signal": "buy" if prob >= 55 else "hold", "last_close": 100.0 + prob,
        "recommendation": "r", "options_source": "yfinance", "options_live": True,
        "generated_at": ts, "data_source": "yahoo",
    }


def _file(dirpath, generated_at, tickers=("AAPL", "MSFT"), models=("lstm_5d", "lstm_jackpot")):
    ts = generated_at.isoformat()
    payload = {
        "schema_version": 1, "generated_at": ts, "producer": "test", "data_source": "yahoo",
        "options_source": "yfinance", "models": list(models), "homepage_tickers": list(tickers),
        "predictions": {m: {t: _pred(t, m, 40.0 + 10 * (i % 6), ts) for i, t in enumerate(tickers)} for m in models},
        "failures": [], "requested": len(tickers), "succeeded": len(tickers), "failed": 0, "truncated": False,
    }
    p = Path(dirpath) / "stock_predictions_latest.json"
    p.write_text(json.dumps(payload))
    return p


def main():
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as d:
        # 1. missing file -> nothing served, status honest
        store = sps.StockPredictionsStore(Path(d) / "stock_predictions_latest.json")
        check(store.get("lstm_5d", "AAPL") is None, "missing file must yield None")
        check(store.available() is False, "missing file must not be available")
        hp = store.homepage(["AAPL"], ["lstm_5d"])
        check(hp["available"] is False and hp["rows"][0]["failures"][0]["error"] == "predictions unavailable", "homepage on missing file")

        # 2. fresh file -> hit, divination shape preserved, alias resolves
        p = _file(d, now - timedelta(hours=2))
        row = store.get("lstm_5d", "aapl")
        check(row is not None and row["cache_status"] == "hit" and "warning" not in row, "fresh row is a hit")
        check(row["probability"] == 40.0 and row["signal"] == "hold" and row["ticker"] == "AAPL", "divination fields preserved")
        check(set(row) >= {"ticker", "model", "description", "horizon", "target_return", "probability", "signal",
                           "last_close", "recommendation", "options_source", "options_live", "generated_at", "data_source"},
              "per-ticker dict keeps every key divination emitted")
        q = store.get("lstm_quant", "MSFT")
        check(q is not None and q["model"] == "lstm_quant" and q["probability"] == 50.0, "lstm_quant aliases lstm_5d")
        check(store.get("lstm_5d", "ZZZZ") is None, "unknown ticker -> None")
        check(store.get("nope", "AAPL") is None, "unknown model -> None")
        st = store.status()
        check(st["loaded"] and st["available"] and st["tickers"] == 2 and st["models"] == ["lstm_5d", "lstm_jackpot"], f"status: {st}")

        # 3. homepage shape == divination's
        hp = store.homepage()
        check(hp["available"] and hp["cache_only"] is True and hp["data_source"] == "yahoo", "homepage header")
        check([r["model"] for r in hp["rows"]] == ["lstm_5d", "lstm_jackpot"], "homepage models default to the file's")
        r0 = hp["rows"][0]
        check(r0["requested_tickers"] == 2 and r0["successful_predictions"] == 2 and r0["failures"] == [], "homepage row counts")
        check(r0["predictions"][0]["cache_status"] == "hit", "homepage predictions carry cache_status")
        hp2 = store.homepage(["AAPL", "NOPE"], ["lstm_jackpot"])
        check(hp2["rows"][0]["successful_predictions"] == 1 and hp2["rows"][0]["failures"] == [{"ticker": "NOPE", "error": "cache-miss"}],
              "explicit tickers/models + cache-miss failures")

        # 4. mtime reload without restart
        time.sleep(0.01)
        _file(d, now - timedelta(hours=1), tickers=("AAPL", "MSFT", "NVDA"))
        os.utime(p, None)
        check(store.status()["tickers"] == 3, "file change picked up via mtime")

        # 5. stale (past STALE_AFTER) but under MAX_AGE -> served with warning
        _file(d, now - timedelta(hours=sps.STOCK_PREDICTIONS_STALE_AFTER_HOURS + 5))
        os.utime(p, None)
        row = store.get("lstm_5d", "AAPL")
        check(row is not None and row["cache_status"] == "stale" and "warning" in row, "stale rows flagged, still served")
        check(store.available(), "stale-but-under-max is still available")

        # 6. past MAX_AGE -> refused
        _file(d, now - timedelta(hours=sps.STOCK_PREDICTIONS_MAX_AGE_HOURS + 1))
        os.utime(p, None)
        check(store.get("lstm_5d", "AAPL") is None and not store.available(), "too-old file refused")
        hp = store.homepage()
        check(hp["available"] is False and hp["rows"][0]["failures"][0]["error"] == "predictions unavailable", "homepage refuses too-old file")

        # 7. corrupt rewrite keeps the last good payload
        _file(d, now - timedelta(hours=1))
        os.utime(p, None)
        check(store.get("lstm_5d", "AAPL") is not None, "good file loads")
        p.write_text("{not json")
        os.utime(p, None)
        check(store.get("lstm_5d", "AAPL") is not None, "corrupt rewrite -> previous payload retained")

        # 8. validator agrees with the store on a fresh file
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ops"))
        import validate_stock_predictions as v  # noqa: E402

        _file(d, now - timedelta(hours=1), tickers=tuple(f"T{i}" for i in range(60)))
        problems, stats = v.validate(str(p), max_age_hours=30, min_tickers=50, min_homepage=5)
        check(problems == [], f"validator accepts a good file: {problems}")
        problems, _ = v.validate(str(p), max_age_hours=0.5)
        check(any("old" in x for x in problems), "validator refuses an old file")
        bad = json.loads(p.read_text())
        for t in bad["predictions"]["lstm_5d"]:
            bad["predictions"]["lstm_5d"][t]["probability"] = 50.0
        p.write_text(json.dumps(bad))
        problems, _ = v.validate(str(p))
        check(any("degenerate" in x for x in problems), "validator refuses constant probabilities")

    print(f"test_stock_predictions_store: {CHECKS} checks passed")


if __name__ == "__main__":
    main()
