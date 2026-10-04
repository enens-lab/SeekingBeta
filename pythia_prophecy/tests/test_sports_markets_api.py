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
            {"type": "moneyline", "season": "2026-27", "recordKind": "record", "graded": 14, "wins": 8, "losses": 6, "pushes": 0},
            {"type": "spread", "season": "2026-27", "recordKind": "calibration", "graded": 30, "wins": None, "losses": None,
             "hits": 16, "expectedHits": 15.2},
            {"type": "total", "season": "2026-27", "graded": 9, "wins": 5, "losses": 4, "pushes": 0}]))
        coll = m.SportsBoardCollection(
            upcoming=[m.SportsUpcomingBoard(id="u1", name="A at B", tour="Football", course="x",
                                            markets=[{"marketId": f"u1:{t}:full_game", "type": t, "label": t, "side": "home",
                                                      "modelProbability": 0.5} for t in ("spread", "total", "team_total_home")])],
            backtests=[m.SportsHistoricalBoard(year=2026, tournament="CAR at ATL", tour="Football", hitStatus="Top Pick",
                                               tournamentId="2026_02_CAR_ATL"),
                       m.SportsHistoricalBoard(year=2026, tournament="X at Y", tour="Football", hitStatus="Miss", tournamentId="other")],
            updated_at=datetime.now(timezone.utc), source="t")
        out = s._attach_sports_markets("football", coll)
        check([r.type for r in out.marketSummary] == ["moneyline"], f"only record rows are served: {out.marketSummary}")
        check(out.backtests[0].markets and out.backtests[0].markets[0].result == "win", "graded pick attached by gameId")
        check(out.backtests[0].hitStatus == "Top Pick" and not out.backtests[1].markets, "hitStatus untouched; no stray attach")
        golf = s._attach_sports_markets("golf", coll)
        check(golf.backtestLabel and golf.backtests == coll.backtests and not golf.marketSummary,
              "sport without market files only gains the simulated-backtest label")
        check(out.backtestLabel and "Simulated" in out.backtestLabel and "betting favourite" in out.backtestLabel,
              "NFL history labelled as the simulated closing-line headline, not the model")
        check(golf.backtestLabel == s.SPORTS_BACKTEST_LABEL, "other sports keep the model backtest label")
        pv = s._preview_sports_board_collection(out, events_cap=1)
        check(len(pv.upcoming[0].markets) == 1, "preview keeps only the headline market")
        dumped = m.SportsBoardsResponse(**{k: out for k in ("golf", "tennis", "basketball", "mlb", "football", "soccer", "olympics")}).model_dump()
        check(dumped["football"]["upcoming"][0]["markets"][0]["type"] == "spread", "markets survive the response model")


def test_history_badges_one_lean_per_type():
    def pk(t, side, p, line=None):
        return {"marketId": f"b:{t}:{side}:{line}", "type": t, "label": f"{t} {side} {line}", "side": side, "line": line,
                "modelProbability": p, "result": "win"}
    picks = [pk("spread", "home", 0.49, -2.5), pk("spread", "away", 0.51, 2.5),
             pk("run_line", "home", 0.36, -1.5), pk("run_line", "away", 0.31, -1.5),
             pk("total", "over", 0.84, 1.5), pk("total", "over", 0.56, 2.5), pk("total", "under", 0.63, 3.5),
             pk("correct_score", "1-1", 0.12), pk("btts", "yes", 0.55)]
    kept = s._history_badge_picks(picks)
    check([(k["type"], k["side"], k["line"]) for k in kept] == [("spread", "away", 2.5), ("total", "over", 2.5), ("btts", "yes", None)],
          f"one lean per type on the most balanced line; nothing under 50%: {kept}")
    # exporter-attached history markets get the same rule
    coll = m.SportsBoardCollection(
        backtests=[m.SportsHistoricalBoard(year=2026, tournament="A at B", tour="Football", hitStatus="Miss",
                                           tournamentId="g1", markets=picks)],
        updated_at=datetime.now(timezone.utc), source="t")
    with tempfile.TemporaryDirectory() as d:
        s.SPORTS_DATA_SOURCE_DIR = Path(d)
        s.SPORTS_FRONTEND_SOURCE_DIR = Path(d)
        s._SPORTS_MARKET_FILES_CACHE.clear()
        out = s._attach_sports_markets("football", coll)
    check(len(out.backtests[0].markets) == 3, "exporter-attached picks filtered too")


def test_season_summary_scope_and_labels():
    def row(date_key, hit, tour):
        return {"year": date_key // 10000, "tournament": f"{tour} {date_key}", "tour": tour, "hitStatus": hit,
                "scheduledDate": date_key, "latestDate": date_key}
    with tempfile.TemporaryDirectory() as d:
        data = Path(d)
        s.SPORTS_DATA_SOURCE_DIR = data
        s.SPORTS_FRONTEND_SOURCE_DIR = data
        s._SPORTS_MARKET_FILES_CACHE.clear()
        (data / "soccer_track_record.json").write_text(json.dumps({"allLeagues": [
            {"tour": "Top 5 leagues", "season": "2026-27", "n": 3, "alwaysHomeAccuracy": 0.432,
             "market": {"n": 3, "favouriteWinRate": 0.52}}]}))
        soccer_rows = [row(20260915, "Top Pick", "Premier League"), row(20260920, "Miss", "La Liga"),
                       row(20260927, "Top Pick", "Serie A"),
                       row(20260705, "Top Pick", "FIFA World Cup"), row(20260712, "Top Pick", "FIFA World Cup")]
        soc = s._sport_season_summary("soccer", soccer_rows)
        cur = 2026 if datetime.now(timezone.utc).month >= 7 else 2025
        if cur == 2026:
            check(soc["sampleSize"] == 3 and soc["topPickHits"] == 2, f"World Cup kept out of the league season: {soc}")
            check(soc["label"] == "Top 5 leagues 2026-27" and "Closing favourite 52.0% (n=3)" in soc["baselineNote"], f"{soc}")
        (data / "basketball_model_record.json").write_text(json.dumps({"records": [
            {"league": "wnba", "season": "2026", "seasonType": "postseason", "alwaysHome": {"accuracy": 0.9}},
            {"league": "wnba", "season": "2026", "seasonType": "regular", "alwaysHome": {"accuracy": 0.5471},
             "elo": {"accuracy": 0.6677},
             "marketMatched": {"games": 331, "model": {"accuracy": 0.6534}, "market": {"accuracy": 0.7043}}}]}))
        s._SPORTS_MARKET_FILES_CACHE.clear()
        check(s._basketball_baseline_note(s._load_market_file("basketball_model_record.json"), "WNBA", "2026")
              == "Always home 54.7% · Elo 66.8% · On 331 games with a closing line: model 65.3% vs closing favourite 70.4%",
              "basketball baselines: regular season row, market on matched games only")
        bb_rows = [row(20260410, "Top Pick", "Basketball"), row(20260612, "Miss", "Basketball"),
                   row(20260820, "Top Pick", "Women's Basketball"), row(20260928, "Top Pick", "Women's Basketball")]
        bb = s._sport_season_summary("basketball", bb_rows)
        check(bb["label"] and bb["label"].startswith("WNBA") and bb["sampleSize"] in (0, 2), f"latest league only: {bb}")
        nfl = s._sport_season_summary("football", [row(20260913, "Top Pick", "Football")])
        check(nfl["basis"] == "market" and nfl["label"].startswith("NFL ") and nfl["baselineNote"] is None, f"{nfl}")
        golf = s._sport_season_summary("golf", [{**row(20260301, "Miss", "PGA"), "rankingFavoriteWon": True},
                                                {**row(20260308, "Top Pick", "PGA"), "rankingFavoriteWon": False}])
        if datetime.now(timezone.utc).year == 2026:
            check(golf["baselineNote"] == "Ranking favourite won 1/2 (50.0%)", f"{golf}")


def test_market_insights_attached_for_the_shown_season():
    with tempfile.TemporaryDirectory() as d:
        data = Path(d)
        s.SPORTS_DATA_SOURCE_DIR = data
        s.SPORTS_FRONTEND_SOURCE_DIR = data
        s._SPORTS_MARKET_FILES_CACHE.clear()
        (data / "basketball_model_record.json").write_text(json.dumps({"leagues": {"wnba": {"walkForwardLines": {
            "2026": {"n": 341, "marginMae": 10.29, "totalMae": 14.95, "marginCoverage80": 0.798}}}}}))
        coll = m.SportsBoardCollection(
            backtests=[m.SportsHistoricalBoard(year=2026, tournament="A at B", tour="Women's Basketball", hitStatus="Miss")],
            updated_at=datetime.now(timezone.utc), source="t",
            seasonSummary=m.SportsBoardSeasonSummary(year=2026, sampleSize=1, topPickHits=0, label="WNBA 2026"))
        out = s._attach_sports_markets("basketball", coll)
        check(len(out.marketInsights) == 3 and out.marketInsights[0].title == "Point spread"
              and out.marketInsights[0].detail.startswith("WNBA 2026"), f"{out.marketInsights}")
        check(out.marketInsightsNote and "Simulated rows" in out.marketInsightsNote, "note attached")
        golf = s._attach_sports_markets("golf", coll.model_copy(update={"marketInsights": []}))
        check(golf.marketInsights == [] and golf.marketInsightsNote is None, "no rows for sports without market data")


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
        numbers = ("year", "sampleSize", "topPickHits", "topPickAccuracy", "top3Hits", "top3Accuracy", "top5Hits", "top5Accuracy")
        if sport != "basketball":   # basketball is now one league's season, not the calendar year
            check(all(getattr(new.seasonSummary, k) == getattr(old.seasonSummary, k) for k in numbers),
                  f"{sport}: season summary numbers changed")
        check(new.seasonSummary.label, f"{sport}: season summary labelled")
        check([b.hitStatus for b in new.backtests] == [b.hitStatus for b in old.backtests], f"{sport}: hitStatus changed")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print(f"test_sports_markets_api: {CHECKS} checks passed")
