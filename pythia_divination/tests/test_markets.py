"""Checks for sports/markets.py (pricing, grading, de-vig).

Run (cwd = pythia_divination):  python tests/test_markets.py
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports import markets as mk  # noqa: E402

STD = 1.0 + 100.0 / 110.0  # -110


def test_grading_hand_computed():
    g = mk.grade_pick
    # spreads (line from the picked side's view)
    assert g("spread", "home", -3.0, 24, 21) == ("push", 0.0)
    assert g("spread", "home", -3.5, 24, 21)[0] == "loss"
    assert g("spread", "away", 3.5, 21, 24) == ("win", round(STD - 1, 4))
    # asian handicap quarter lines
    assert g("asian_handicap", "home", -0.75, 2, 1)[0] == "half_win"
    assert g("asian_handicap", "home", -0.25, 1, 1) == ("half_loss", -0.5)
    assert g("asian_handicap", "away", 0.25, 1, 1)[0] == "half_win"
    assert g("asian_handicap", "home", -1.25, 3, 1)[0] == "win"
    assert g("asian_handicap", "home", -1.25, 2, 1)[0] == "half_loss"
    # totals incl. quarter
    assert g("total", "over", 44.5, 24, 21)[0] == "win"
    assert g("total", "under", 45.0, 24, 21) == ("push", 0.0)
    assert g("total", "over", 2.75, 2, 1)[0] == "half_win"
    assert g("total", "under", 2.25, 1, 1)[0] == "half_win"   # push on 2.0, win on 2.5
    assert g("total", "under", 2.25, 1, 0)[0] == "win"
    assert g("total", "under", 2.25, 2, 0)[0] == "half_win"
    assert g("total", "under", 2.25, 2, 1)[0] == "loss"
    assert g("team_total_home", "over", 3.5, 4, 0)[0] == "win"
    # moneyline / 1x2 / btts / double chance / dnb
    assert g("moneyline", "home", None, 20, 20) == ("push", 0.0)
    assert g("1x2", "draw", None, 1, 1)[0] == "win"
    assert g("btts", "yes", None, 1, 0)[0] == "loss"
    assert g("double_chance", "home_draw", None, 1, 1)[0] == "win"
    assert g("draw_no_bet", "away", None, 0, 0) == ("push", 0.0)


def test_pricing_agrees_with_grading_on_every_score():
    """For a score matrix, the priced outcome mix must equal the probability-weighted
    mix of grade_pick over all scores -- the two code paths cannot drift apart."""
    rng = np.random.default_rng(7)
    lam_h, lam_a = 1.55, 1.05
    i = np.arange(11)
    ph = np.exp(-lam_h) * lam_h ** i / np.array([math.factorial(k) for k in i])
    pa = np.exp(-lam_a) * lam_a ** i / np.array([math.factorial(k) for k in i])
    m = np.outer(ph, pa); m /= m.sum()
    for side in ("home", "away"):
        for line in (-1.75, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.25):
            priced = mk.score_matrix_handicap(m, side, line).as_dict()
            brute = {k: 0.0 for k in mk.OUTCOMES}
            for a in range(11):
                for b in range(11):
                    brute[mk.grade_pick("asian_handicap", side, line, a, b)[0]] += m[a, b]
            for k in mk.OUTCOMES:
                assert abs(priced[k] - brute[k]) < 1e-3, (side, line, k, priced[k], brute[k])
    for side in ("over", "under"):
        for line in (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5):
            priced = mk.score_matrix_total(m, side, line).as_dict()
            brute = {k: 0.0 for k in mk.OUTCOMES}
            for a in range(11):
                for b in range(11):
                    brute[mk.grade_pick("total", side, line, a, b)[0]] += m[a, b]
            for k in mk.OUTCOMES:
                assert abs(priced[k] - brute[k]) < 1e-3, (side, line, k, priced[k], brute[k])
    hm = mk.score_matrix_markets(m)
    assert abs(hm["home"] + hm["draw"] + hm["away"] - 1) < 1e-9
    assert abs(hm["exp_home_goals"] - lam_h) < 0.01 and abs(hm["exp_away_goals"] - lam_a) < 0.01
    _ = rng


def test_normal_spread_and_total_pricing():
    # a pick-em game: 50/50 at 0 ex-push, with push mass on a whole line
    p0 = mk.spread_prices_normal(0.0, 13.5, 0.0)
    assert abs(p0.win - p0.loss) < 1e-6 and p0.push > 0.02
    # a 3-point favourite at -3: push mass, P(cover ex push) about 0.5
    p3 = mk.spread_prices_normal(3.0, 13.5, -3.0)
    assert abs(p3.win_ex_push - 0.5) < 0.01 and p3.push > 0.02
    # half-point lines never push; symmetry between sides
    ph = mk.spread_prices_normal(6.0, 13.5, -3.5)
    pa = mk.spread_prices_normal(-6.0, 13.5, 3.5)
    assert ph.push == 0 and abs(ph.win - pa.loss) < 1e-9 and ph.win_ex_push > 0.55
    # totals: mean 45 at 44.5 is a slight over; over+under cover the space
    o = mk.total_prices_normal(45.0, 10.0, 44.5, "over")
    u = mk.total_prices_normal(45.0, 10.0, 44.5, "under")
    assert o.win > 0.5 and abs(o.win - u.loss) < 1e-9 and abs(o.win + u.win - 1) < 1e-9
    assert mk.fair_line_from_mean(44.2) == 44.5 and mk.fair_line_from_mean(-3.7) == -3.5


def test_odds_helpers():
    assert abs(mk.american_to_decimal(-110) - 1.9091) < 1e-3
    assert mk.american_to_decimal(150) == 2.5
    assert mk.decimal_to_american(2.5) == 150 and mk.decimal_to_american(1.5) == -200
    p = mk.devig([1.91, 1.91]); assert abs(p[0] - 0.5) < 1e-9
    p3 = mk.devig([2.1, 3.4, 3.6]); assert abs(sum(p3) - 1) < 1e-9 and p3[0] > p3[2]
    assert mk.probability_to_fair_american(0.6) == -150


def test_market_pick_shape_and_edge_gate():
    prices = mk.spread_prices_normal(4.8, 13.5, -3.5)
    market = {"line": -3.5, "americanOdds": -110, "decimalOdds": 1.909, "impliedProbability": 0.5, "source": "test"}
    hidden = mk.market_pick(board_id="b1", market_type="spread", side="home", label="KC -3.5", prices=prices,
                            line=-3.5, model_line=-4.5, market=market, show_edge=False)
    shown = mk.market_pick(board_id="b1", market_type="spread", side="home", label="KC -3.5", prices=prices,
                           line=-3.5, model_line=-4.5, market=market, show_edge=True)
    assert hidden["marketId"] == "b1:spread:full_game" and "edge" not in hidden
    assert "edge" in shown and abs(shown["edge"] - (shown["modelProbability"] - 0.5)) < 1e-9
    assert hidden["confidenceTier"] in ("low", "medium", "high")
    assert set(hidden["outcomeProbabilities"]) == set(mk.OUTCOMES)


def test_real_asian_handicap_outcome_mix():
    """Grade every closing AH line in the local football-data files; the analyst
    measured win 37.72 / half_win 6.69 / push 6.18 / half_loss 8.24 / loss 41.17 (%)."""
    import pandas as pd
    files = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "data/sports/soccer/raw/*_2[3-5]2[4-6].csv")))
    if not files:
        print("  (skipped: no local football-data files)")
        return
    counts = {k: 0 for k in mk.OUTCOMES}
    n = 0
    for f in files:
        df = pd.read_csv(f, encoding="latin-1")
        col = "AHCh" if "AHCh" in df.columns else "AHh"
        for row in df[["FTHG", "FTAG", col]].dropna().itertuples(index=False):
            counts[mk.grade_pick("asian_handicap", "home", float(row[2]), row[0], row[1])[0]] += 1
            n += 1
    share = {k: round(100 * v / n, 2) for k, v in counts.items()}
    print("  closing AH home-side outcome mix over %d matches: %s" % (n, share))
    assert n > 3000
    assert 30 < share["win"] < 45 and share["half_win"] > 3 and share["push"] > 3 and share["half_loss"] > 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_markets: all checks passed")
