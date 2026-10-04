"""Plain-language market rows for the sports History tab (api/market_insights.py).

Run (cwd = pythia_prophecy):  python tests/test_market_insights.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import market_insights as mi  # noqa: E402

JARGON = re.compile(r"log-?loss|brier|calibrat|\bmae\b|rps|ats\b|units|roi\b|\bvig|edge|lock|value bet", re.I)

BASKETBALL = {"leagues": {"wnba": {"walkForwardLines": {
    "2025": {"n": 311, "marginMae": 10.69, "totalMae": 13.27, "marginCoverage80": 0.778},
    "2026": {"n": 341, "marginMae": 10.29, "totalMae": 14.95, "marginCoverage80": 0.798},
    "postseason": {"n": 34, "marginMae": 11.07, "totalMae": 12.75, "marginCoverage80": 0.765}}}}}
SOCCER = {"allLeagues": [{
    "tour": "Top 5 leagues", "season": "2026-27", "n": 250,
    "overUnder25": {"modelLogLoss": 0.6624, "baseRateLogLoss": 0.6776, "marketLogLoss": 0.6409},
    "btts": {"modelLogLoss": 0.6763, "baseRateLogLoss": 0.6839},
    "correctScoreTop3HitRate": 0.304, "naiveCorrectScoreTop3HitRate": 0.26,
    "meanGoals": {"predicted": 2.808, "actual": 3.044}}]}


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_basketball_rows_use_the_shown_league_and_season():
    rows = mi.build_market_insights("basketball", summary=[], record=BASKETBALL, league="WNBA", season="2026")
    heads = [r["headline"] for r in rows]
    check(heads == ["Our fair spread was off by 10.3 points per game on average",
                    "Our fair total was off by 14.9 points per game on average",
                    "The final margin landed inside our 80% range in 80% of games"], f"{heads}")
    check(all(r["detail"] == "WNBA 2026 · 341 games · simulated" and r["basis"] == "simulated" for r in rows), f"{rows}")
    check(rows[2]["comparison"] == "A well-sized range lands close to 80%", "range explained in words")
    # unknown season falls back to the latest regular season, never the postseason
    rows = mi.build_market_insights("basketball", summary=[], record=BASKETBALL, league="WNBA", season="2027")
    check(rows[0]["detail"].startswith("WNBA 2026"), f"{rows[0]}")


def test_soccer_rows_compare_in_words():
    rows = {r["title"]: r for r in mi.build_market_insights("soccer", summary=[], record=SOCCER, season="2026-27")}
    check(rows["Over/under 2.5 goals"]["headline"] ==
          "Our probabilities were closer to the results than the league average, but not as close as the betting market",
          rows["Over/under 2.5 goals"]["headline"])
    check(rows["Both teams to score"]["headline"] == "Our probabilities were closer to the results than the league average", "btts")
    check(rows["Exact score"]["headline"] == "The real score was one of our top 3 in 30% of matches"
          and rows["Exact score"]["comparison"] == "Always guessing 1-0, 1-1 and 2-1: 26%", f"{rows['Exact score']}")
    check(rows["Goals per match"]["headline"] == "We expected 2.8 goals per match; there were 3.0", "goals")
    check(rows["Exact score"]["detail"] == "Top 5 leagues 2026-27 · 250 matches · simulated", "detail")
    worse = mi._closeness(0.70, 0.69, 0.66)
    check(worse == "Our probabilities were not closer to the results than the league average, but not as close as the betting market", worse)


def test_live_rows_compare_expected_with_actual_and_hide_small_samples():
    summary = [
        {"type": "total", "season": "2026", "graded": 49, "pushes": 0, "hits": 25, "expectedHits": 23.05, "recordKind": "calibration"},
        {"type": "run_line", "season": "2026", "graded": 8, "hits": 2, "expectedHits": 2.76, "recordKind": "calibration"},
        {"type": "spread", "season": "2026", "graded": 52, "pushes": 3, "hits": 25, "expectedHits": 24.4, "recordKind": "calibration"},
        {"type": "moneyline", "season": "2026", "graded": 120, "wins": 66, "losses": 54, "pushes": 0,
         "winRateExPush": 0.55, "breakEvenRate": 0.5238, "recordKind": "record"},
        {"type": "player_prop", "season": "2026", "graded": 500, "hits": 1, "expectedHits": 1},
    ]
    rows = mi.live_rows(summary, "2026")
    titles = [r["title"] for r in rows]
    check(titles == ["Moneyline", "Point spread", "Total (over/under)"], f"display order, small samples and unknown types hidden: {titles}")
    by = {r["title"]: r for r in rows}
    check(by["Total (over/under)"]["headline"] == "Our probabilities said about 23 of 49 would win; 25 did", by["Total (over/under)"]["headline"])
    check(by["Point spread"]["headline"] == "Our probabilities said about 24 of 49 would win; 25 did", "pushes excluded from the count")
    check(by["Total (over/under)"]["detail"] == "2026 · 49 graded (early) · live", by["Total (over/under)"]["detail"])
    check(by["Moneyline"]["headline"] == "Won 66 of 120" and by["Moneyline"]["comparison"] == "55.0% won; 52.4% needed to break even at these prices",
          f"{by['Moneyline']}")
    check(by["Moneyline"]["detail"] == "2026 · 120 graded · live", "no early tag at 100+")


def test_no_jargon_anywhere():
    rows = (mi.build_market_insights("basketball", summary=[], record=BASKETBALL, league="WNBA", season="2026")
            + mi.build_market_insights("soccer", summary=[], record=SOCCER, season="2026-27")
            + mi.live_rows([{"type": "total", "season": "2026", "graded": 49, "hits": 25, "expectedHits": 23}], "2026"))
    text = " ".join(str(v) for r in rows for v in r.values() if v) + " " + mi.INSIGHTS_NOTE
    check(not JARGON.search(text), f"jargon found: {JARGON.findall(text)}")


def test_missing_or_bad_inputs_return_nothing():
    check(mi.build_market_insights("mlb", summary=None, record=None) == [], "mlb without live data")
    check(mi.build_market_insights("basketball", summary="x", record={"leagues": "bad"}, league="NBA") == [], "bad shapes")
    check(mi.build_market_insights("soccer", summary=[], record={"allLeagues": []}) == [], "empty soccer")


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print("ok", name)
    print(f"test_market_insights: {n} tests passed")
