"""Checks for the markets plumbing in api/service.py.

Run:  python tests/test_sports_markets_api.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("JWT_SECRET_KEY", "test")

from api import service as s  # noqa: E402
from api import models as m  # noqa: E402

CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def _bt(date_key: int, hit: str, tid: str = "", year: int | None = None) -> dict:
    return {"year": year or date_key // 10000, "tournament": f"G {date_key} {tid}", "tour": "Football",
            "hitStatus": hit, "scheduledDate": date_key, "latestDate": date_key, "tournamentId": tid}


def test_season_keying_cross_year():
    # Jan 2026 games belong to the 2025 NFL season; Sep 2026 to the 2026 season.
    rows = [_bt(20260111, "Top Pick"), _bt(20260118, "Miss"), _bt(20260913, "Top Pick"), _bt(20260920, "Top Pick"), _bt(20260927, "Miss")]
    nfl = s._build_sports_season_summary(rows, current_year=2026, season_start_month=8)
    check(nfl["year"] == 2026 and nfl["sampleSize"] == 3 and nfl["topPickHits"] == 2, f"NFL 2026 season: {nfl}")
    nfl25 = s._build_sports_season_summary(rows, current_year=2025, season_start_month=8)
    check(nfl25["sampleSize"] == 2, f"NFL 2025 season includes January games: {nfl25}")
    cal = s._build_sports_season_summary(rows, current_year=2026)
    check(cal["sampleSize"] == 5, "calendar keying unchanged for other sports")
    # compact summary rows (key tuple) are keyed by their date too
    compact = [s._summary_row(r) for r in rows]
    check(s._build_sports_season_summary(compact, current_year=2026, season_start_month=8)["sampleSize"] == 3, "compact rows")


def test_market_files_attach_and_preview_trim():
    with tempfile.TemporaryDirectory() as d:
        data = Path(d)
        s.SPORTS_DATA_SOURCE_DIR = data
        s.SPORTS_FRONTEND_SOURCE_DIR = data
        s._SPORTS_MARKET_FILES_CACHE.clear()
        pick = {"marketId": "football-2026_02_CAR_ATL:spread:full_game", "boardId": "football-2026_02_CAR_ATL",
                "gameId": "2026_02_CAR_ATL", "type": "spread", "label": "ATL -2.5", "side": "home", "line": -2.5,
                "modelProbability": 0.51, "result": "win", "unitReturn": 0.9091}
        (data / "football_market_history.json").write_text(json.dumps([pick]))
        (data / "football_market_summary.json").write_text(json.dumps([
            {"type": "spread", "season": "2026-27", "graded": 14, "wins": 8, "losses": 6, "pushes": 0}]))
        coll = m.SportsBoardCollection(
            upcoming=[m.SportsUpcomingBoard(id="u1", name="A at B", tour="Football", course="x",
                                            markets=[{"marketId": f"u1:{t}:full_game", "type": t, "label": t, "side": "home",
                                                      "modelProbability": 0.5} for t in ("spread", "total", "team_total_home")])],
            backtests=[m.SportsHistoricalBoard(year=2026, tournament="CAR at ATL", tour="Football", hitStatus="Top Pick",
                                               tournamentId="2026_02_CAR_ATL"),
                       m.SportsHistoricalBoard(year=2026, tournament="X at Y", tour="Football", hitStatus="Miss", tournamentId="other")],
            updated_at=datetime.now(timezone.utc), source="t")
        out = s._attach_sports_markets("football", coll)
        check(len(out.marketSummary) == 1 and out.marketSummary[0].season == "2026-27", "summary attached")
        check(out.backtests[0].markets and out.backtests[0].markets[0].result == "win", "graded pick attached by gameId")
        check(out.backtests[0].hitStatus == "Top Pick" and not out.backtests[1].markets, "hitStatus untouched; no stray attach")
        check(s._attach_sports_markets("golf", coll) is coll, "sport without market files is untouched")
        pv = s._preview_sports_board_collection(out, events_cap=1)
        check(len(pv.upcoming[0].markets) == 1, "preview keeps only the headline market")
        dumped = m.SportsBoardsResponse(**{k: out for k in ("golf", "tennis", "basketball", "mlb", "football", "soccer", "olympics")}).model_dump()
        check(dumped["football"]["upcoming"][0]["markets"][0]["type"] == "spread", "markets survive the response model")


def test_current_files_unchanged_for_other_sports():
    """hitStatus and seasonSummary on the repo's current board files must be
    identical to the pre-markets code path for sports without a season change."""
    data_dir = Path(__file__).resolve().parents[1] / "frontend" / "src" / "data"
    s.SPORTS_DATA_SOURCE_DIR = data_dir
    s.SPORTS_FRONTEND_SOURCE_DIR = data_dir
    s._SPORTS_BACKTESTS_CACHE.clear()
    for sport, fname in (("mlb", "mlb_historical_backtests.json"), ("basketball", "basketball_historical_backtests.json"),
                         ("tennis", "wta_historical_backtests.json"), ("golf", "historical_backtests.json")):
        if not (data_dir / fname).exists():
            continue
        full = s._load_sports_json(fname)
        kept, rows = s._load_sports_backtests(fname)
        new = s._build_sports_board_collection(upcoming=[], backtests=kept, updated_at=datetime.now(timezone.utc),
                                               source="t", summary_rows=rows, sport=sport)
        old = s._build_sports_board_collection(upcoming=[], backtests=full, updated_at=datetime.now(timezone.utc), source="t")
        check(new.seasonSummary == old.seasonSummary, f"{sport}: season summary changed")
        check([b.hitStatus for b in new.backtests] == [b.hitStatus for b in old.backtests], f"{sport}: hitStatus changed")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print(f"test_sports_markets_api: {CHECKS} checks passed")
