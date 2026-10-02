"""Checks for sports/market_log.py (append-only pick log + grading from it).

Run (cwd = pythia_divination):  python tests/test_market_log.py
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports import market_log as ml  # noqa: E402
from sports import markets as mk  # noqa: E402


def _board(line: float, prob_margin: float, gid: str = "2026_05_KC_BUF", date: int = 20261011) -> dict:
    prices = mk.spread_prices_normal(prob_margin, 13.5, line)
    pick = mk.market_pick(board_id=f"football-{gid}", market_type="spread", side="home", label=f"BUF {line:+}",
                          prices=prices, line=line, model_line=-round(prob_margin * 2) / 2)
    total = mk.market_pick(board_id=f"football-{gid}", market_type="total", side="over", label="Over 47.5",
                           prices=mk.total_prices_normal(49.0, 10.0, 47.5, "over"), line=47.5, model_line=49.0)
    return {"id": f"football-{gid}", "gameId": gid, "scheduledDate": date, "gameStart": "2026-10-11T17:00:00Z",
            "homeTeam": "BUF", "awayTeam": "KC", "markets": [pick, total]}


def test_log_select_grade_summarize():
    season = ml.season_cross_year(8)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Thursday bake: home -2.5 ; Sunday morning bake: line moved to -3.5 ; a post-kickoff bake must be ignored
        recs1 = ml.picks_from_boards("football", [_board(-2.5, 4.0)], season_of=season, model_version="v1",
                                     published_at="2026-10-08T07:12:00+00:00")
        recs2 = ml.picks_from_boards("football", [_board(-3.5, 4.4)], season_of=season, model_version="v1",
                                     published_at="2026-10-11T07:12:00+00:00")
        recs3 = ml.picks_from_boards("football", [_board(-7.5, 9.0)], season_of=season, model_version="v2",
                                     published_at="2026-10-11T20:00:00+00:00")
        for i, recs in enumerate((recs1, recs2, recs3)):
            ml.write_snapshot("football", recs, root=root, now=datetime(2026, 10, 8 + i, 7, 12, tzinfo=timezone.utc))
        files = sorted((root / "football").glob("picks_*.jsonl"))
        assert len(files) == 3, files
        # write-once: same timestamp gets a suffix, never overwrites
        ml.write_snapshot("football", recs1, root=root, now=datetime(2026, 10, 8, 7, 12, tzinfo=timezone.utc))
        assert len(list((root / "football").glob("picks_*.jsonl"))) == 4

        snaps = ml.load_snapshots("football", root=root)
        chosen = ml.pregame_picks(snaps)
        spread = chosen["football-2026_05_KC_BUF:spread:full_game"]
        assert spread["line"] == -3.5 and spread["modelVersion"] == "v1", spread  # last PRE-game snapshot, not the post-kickoff v2
        assert spread["season"] == "2026-27" and spread["gameDate"] == 20261011

        graded = ml.grade_picks(chosen, {"2026_05_KC_BUF": (27.0, 24.0)})  # BUF by 3: -3.5 loses, over 47.5 wins (51)
        by_type = {g["type"]: g for g in graded}
        assert by_type["spread"]["result"] == "loss" and by_type["total"]["result"] == "win"
        assert ml.grade_picks(chosen, {}) == []  # no final score -> ungraded, not voided


def test_summary_math_and_small_sample_hiding():
    graded = []
    for r in ["win"] * 20 + ["loss"] * 15 + ["push"] * 3 + ["half_win"] * 2 + ["half_loss"] * 2:
        unit = {"win": 0.9091, "half_win": 0.4545, "push": 0.0, "half_loss": -0.5, "loss": -1.0}[r]
        graded.append({"type": "spread", "season": "2026-27", "result": r, "unitReturn": unit, "modelProbability": 0.55,
                       "market": {"impliedProbability": 0.5, "decimalOdds": 1.9091}})
    graded += [{"type": "total", "season": "2026-27", "result": "win", "unitReturn": 0.9091, "modelProbability": 0.6}] * 5
    s = {x["type"]: x for x in ml.summarize(graded)}
    sp = s["spread"]
    assert sp["graded"] == 42 and sp["wins"] == 21.0 and sp["losses"] == 16.0 and sp["pushes"] == 3
    assert abs(sp["winRateExPush"] - 21 / 37) < 1e-3
    assert abs(sp["breakEvenRate"] - 0.5238) < 1e-3
    assert abs(sp["unitsAtStatedPrice"] - (20 * 0.9091 - 15 + 2 * 0.4545 - 1.0)) < 0.02
    assert sp["brier"] is not None and sp["marketBrier"] == 0.25
    # 5 graded totals: below the minimum sample -> no rate, no ROI shown
    assert s["total"]["winRateExPush"] is None and s["total"]["roi"] is None and s["total"]["graded"] == 5


def test_season_keys():
    nfl = ml.season_cross_year(8)
    assert nfl(20260913) == "2026-27" and nfl(20270207) == "2026-27" and nfl(20260601) == "2025-26"
    assert ml.season_calendar(20261003) == "2026"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_market_log: all checks passed")
