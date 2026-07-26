#!/usr/bin/env python3
"""Reject degenerate sports board exports before they reach users.

Context: on 2026-07-26 the football and basketball exporters were found to have
recorded the AWAY team as winner in 100% of games (856 of them), because
``getattr(row, "home_win", 0)`` turned a missing label into a fabricated result.
Every published accuracy figure for those sports was wrong for months and nothing
caught it, because corrupt data looks exactly like real data to a JSON sync.

This validator is the tripwire. It runs after the RunPod boards sync and before
prophecy-api is rebuilt, so a bad export is refused rather than baked and served.
It lives in the EC2 repo (not the worker image), so it protects against a stale
worker still running old exporter code.

The check: for each *_historical_backtests.json with enough gradeable games, the
home-team win rate must be plausible. Real rates cluster near 54-60% (NFL, NBA,
MLB, NHL). A rate of 0% or 100% is arithmetically impossible across hundreds of
games and means the winner field is being derived, not read.

Usage:
    python3 ops/validate_sports_boards.py <data-dir>
Exit 0 = all good (or too little data to judge). Exit 1 = degenerate, do not ship.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Below this many gradeable games, a skewed rate is plausible noise, not corruption.
MIN_SAMPLE = 30
# Deliberately wide. This catches gross corruption (0%/100%), not modelling nuance.
MIN_HOME_RATE = 0.30
MAX_HOME_RATE = 0.80


def audit_file(path: Path) -> tuple[str, int, float | None, str]:
    """(label, graded, home_rate, verdict) for one backtests file."""
    label = path.name.replace("_historical_backtests.json", "")
    try:
        rows = json.loads(path.read_text())
    except Exception as exc:
        return label, 0, None, f"UNREADABLE ({type(exc).__name__})"
    if not isinstance(rows, list):
        return label, 0, None, "SKIP (not a list)"

    home = away = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        winner = row.get("actualWinner")
        home_team, away_team = row.get("homeTeam"), row.get("awayTeam")
        if not winner or not home_team or not away_team:
            continue
        if winner == home_team:
            home += 1
        elif winner == away_team:
            away += 1

    graded = home + away
    if graded < MIN_SAMPLE:
        # Ranked-field sports (golf, tennis) and draw-capable ones (soccer) do not
        # express results as home/away, so they legitimately land here.
        return label, graded, None, "SKIP (too few home/away games to judge)"

    rate = home / graded
    if rate < MIN_HOME_RATE or rate > MAX_HOME_RATE:
        return label, graded, rate, "DEGENERATE"
    return label, graded, rate, "OK"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_sports_boards.py <data-dir>", file=sys.stderr)
        return 2
    data_dir = Path(sys.argv[1])
    files = sorted(data_dir.glob("*_historical_backtests.json"))
    if not files:
        print(f"[validate] no backtests files in {data_dir}; nothing to check")
        return 0

    failures = []
    for path in files:
        label, graded, rate, verdict = audit_file(path)
        rate_text = f"{rate:.0%}" if rate is not None else "n/a"
        print(f"[validate] {label:12s} graded={graded:<5d} home_rate={rate_text:>5s}  {verdict}")
        if verdict in {"DEGENERATE", "UNREADABLE"}:
            failures.append((label, verdict, rate_text, graded))

    if failures:
        print()
        print("[validate] REFUSING TO SHIP. Implausible home-win rates mean the")
        print("[validate] winner field is being fabricated, not read:")
        for label, verdict, rate_text, graded in failures:
            print(f"[validate]   {label}: {verdict} (home_rate={rate_text} over {graded} games)")
        print("[validate] Expected roughly 54-60% for NFL/NBA/MLB/NHL.")
        print("[validate] Likely cause: an exporter reading a label that a pandas")
        print("[validate] merge renamed (home_win -> home_win_x/_y), or a source")
        print("[validate] table carrying only scores. See MARKET_ROADMAP.md.")
        print("[validate] If the RunPod worker image predates that fix, rebuild it.")
        return 1

    print("[validate] all sports within plausible bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
