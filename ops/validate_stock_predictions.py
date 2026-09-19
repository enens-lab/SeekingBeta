#!/usr/bin/env python3
"""Tripwire for the daily stock-prediction sweep before it goes live.

Runs on the box (host python3, no deps) in ops/refresh_stock_predictions_runpod.sh
between "pulled from S3" and "moved into the served path". Refuses a file that would
make every /predict/* call fail or serve nonsense:

  * parseable JSON with schema_version 1 and a `predictions` map
  * generated_at parses and is at most --max-age-hours old (default 30: the run
    that just finished, not a stale re-upload)
  * every requested model is present with at least --min-tickers rows
  * the homepage tickers are present for lstm_5d and lstm_jackpot (the carousel and
    the Daily Brief read exactly those; --require-homepage to enforce all of them,
    default: at least 5 of 7)
  * probabilities are numeric within [0, 100], last_close > 0, signal in the known
    set; no more than --max-bad-fraction of rows may fail this
  * the sweep is not degenerate: lstm_5d probabilities are not all the same value
    (a scaler/feature bug produces a constant), and not all options_live=false when
    the run claims a live options source

Usage: python3 ops/validate_stock_predictions.py <file> [--max-age-hours 30] [--min-tickers 50]
Exit 0 = safe to serve, 1 = refuse (details on stderr).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

SIGNALS = {"strong_buy", "buy", "hold", "avoid", "sell"}
REQUIRED_MODELS = ("lstm_5d", "lstm_jackpot")


def _parse_ts(value):
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def validate(path, max_age_hours=30.0, min_tickers=50, min_homepage=5, max_bad_fraction=0.02,
             require_homepage=False):
    problems = []
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return [f"unreadable JSON: {exc}"], {}
    if not isinstance(d, dict) or d.get("schema_version") != 1:
        problems.append(f"schema_version != 1 (got {d.get('schema_version') if isinstance(d, dict) else type(d).__name__})")
    preds = d.get("predictions") if isinstance(d, dict) else None
    if not isinstance(preds, dict):
        return problems + ["no 'predictions' map"], {}

    ts = _parse_ts(d.get("generated_at"))
    if ts is None:
        problems.append(f"generated_at unparseable: {d.get('generated_at')!r}")
    else:
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        if age_h > max_age_hours:
            problems.append(f"generated_at is {age_h:.1f}h old (> {max_age_hours}h)")
        if age_h < -1:
            problems.append(f"generated_at is {-age_h:.1f}h in the future")

    stats = {"generated_at": d.get("generated_at"), "models": {}}
    for model in REQUIRED_MODELS:
        table = preds.get(model)
        if not isinstance(table, dict):
            problems.append(f"model {model} missing")
            continue
        n = len(table)
        bad = 0
        probs = []
        live = 0
        for ticker, row in table.items():
            if not isinstance(row, dict):
                bad += 1
                continue
            p = row.get("probability")
            lc = row.get("last_close")
            ok = (isinstance(p, (int, float)) and 0.0 <= float(p) <= 100.0
                  and isinstance(lc, (int, float)) and float(lc) > 0
                  and str(row.get("signal")) in SIGNALS
                  and str(row.get("ticker", "")).upper() == str(ticker).upper())
            if not ok:
                bad += 1
            else:
                probs.append(round(float(p), 2))
            if row.get("options_live"):
                live += 1
        stats["models"][model] = {"tickers": n, "bad_rows": bad, "options_live": live}
        if n < min_tickers:
            problems.append(f"{model}: only {n} tickers (< {min_tickers})")
        if n and bad / n > max_bad_fraction:
            problems.append(f"{model}: {bad}/{n} malformed rows")
        if len(probs) >= 10 and len(set(probs)) <= 2:
            problems.append(f"{model}: degenerate probabilities (only {len(set(probs))} distinct values)")
        if n >= 20 and live == 0 and str(d.get("options_source", "")).lower() in ("yfinance", "schwab", "auto"):
            problems.append(f"{model}: options_live=false for every ticker although options_source={d.get('options_source')}")

    homepage = [str(t).upper() for t in (d.get("homepage_tickers") or [])]
    if homepage:
        for model in REQUIRED_MODELS:
            table = preds.get(model) or {}
            present = [t for t in homepage if t in table]
            stats["models"].setdefault(model, {})["homepage_present"] = len(present)
            need = len(homepage) if require_homepage else min(min_homepage, len(homepage))
            if len(present) < need:
                problems.append(f"{model}: homepage coverage {len(present)}/{len(homepage)} (< {need}); missing {sorted(set(homepage) - set(present))}")
    else:
        problems.append("homepage_tickers missing/empty")

    if d.get("truncated"):
        stats["truncated"] = True  # informational: partial but valid
    return problems, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-age-hours", type=float, default=30.0)
    ap.add_argument("--min-tickers", type=int, default=50)
    ap.add_argument("--min-homepage", type=int, default=5)
    ap.add_argument("--max-bad-fraction", type=float, default=0.02)
    ap.add_argument("--require-homepage", action="store_true")
    args = ap.parse_args()
    problems, stats = validate(args.path, args.max_age_hours, args.min_tickers, args.min_homepage,
                               args.max_bad_fraction, args.require_homepage)
    print(f"[validate-stock-predictions] {args.path}: {json.dumps(stats, sort_keys=True)}")
    if problems:
        for p in problems:
            print(f"[validate-stock-predictions] REFUSE: {p}", file=sys.stderr)
        return 1
    print("[validate-stock-predictions] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
