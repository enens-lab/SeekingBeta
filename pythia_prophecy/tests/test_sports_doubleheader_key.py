"""MLB doubleheaders must not collapse into one board when histories are merged.

Both games of a doubleheader share (tour, "Away at Home", date), so the backtest key
also carries the game id for baseball; other tours keep the (tour, name, date) key.

Run:  python tests/test_sports_doubleheader_key.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JWT_SECRET_KEY", "test")

from api import service as s  # noqa: E402


def _mlb(game_id: str, hit: str) -> dict:
    return {"tour": "Baseball", "tournament": "Boston Red Sox at New York Yankees", "scheduledDate": 20250705,
            "latestDate": 20250705, "year": 2025, "tournamentId": f"mlb-{game_id}", "gameId": game_id, "hitStatus": hit}


def main() -> None:
    game1, game2 = _mlb("777001", "Top Pick"), _mlb("777002", "Miss")
    assert s._sports_backtest_key(game1) != s._sports_backtest_key(game2)

    merged = s._merge_runtime_backtests([game1], [game2])
    assert {b["gameId"] for b in merged} == {"777001", "777002"}, merged
    # the runtime copy of the same game still replaces the static one
    regraded = {**game1, "hitStatus": "Miss"}
    merged = s._merge_runtime_backtests([game1, game2], [regraded])
    assert len(merged) == 2 and next(b for b in merged if b["gameId"] == "777001")["hitStatus"] == "Miss"

    rows = s._merge_summary_rows([s._summary_row(game1)], [game2])
    assert len(rows) == 2 and sorted(r["hitStatus"] for r in rows) == ["Miss", "Top Pick"]

    # other tours: unchanged (tour, name, date) identity, game ids ignored
    a = {"tour": "NFL", "tournament": "Bills at Jets", "scheduledDate": 20250907, "gameId": "x1"}
    b = {**a, "gameId": "x2"}
    assert s._sports_backtest_key(a) == s._sports_backtest_key(b)
    assert len(s._merge_runtime_backtests([a], [b])) == 1
    print("test_sports_doubleheader_key: all checks passed")


if __name__ == "__main__":
    main()
