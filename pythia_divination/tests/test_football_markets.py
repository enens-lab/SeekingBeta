"""Checks for the NFL market-anchored win probability, ladders and model view.

Run (cwd = pythia_divination):  python tests/test_football_markets.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports import market_log as ml  # noqa: E402
from sports import markets as mk  # noqa: E402
from sports.football import markets_nfl as nfl  # noqa: E402
from sports.football import model_view as mv  # noqa: E402
from sports.football.build_training_dataset import _LEAKY_COLUMNS  # noqa: E402
from sports.football.feature_engineering import build_team_game_logs, prepare_games  # noqa: E402

PARAMS = mv.load_params()
LADDER = (PARAMS or {}).get("ladder") or {}

# nflverse convention: spread_line > 0 means the HOME team is favoured.
GAME = {
    "home_team": "BUF", "away_team": "NE", "home_moneyline": -298, "away_moneyline": 240,
    "spread_line": 7.0, "home_spread_odds": -105, "away_spread_odds": -115,
    "total_line": 48.5, "over_odds": -115, "under_odds": -105,
}
MODEL = {"home_win_probability": 0.68, "margin": 6.2, "total": 47.1}


def _board(game=GAME, model=MODEL, ladder=LADDER, gid="2026_04_NE_BUF"):
    markets = nfl.build_board_markets(board_id=f"football-{gid}", game=game, ladder=ladder, model=model,
                                      published_at="2026-10-02T07:10:00+00:00", captured_at="2026-10-02T07:00:00Z",
                                      model_version="test-v1")
    return {"id": f"football-{gid}", "gameId": gid, "scheduledDate": 20261004, "gameStart": "2026-10-04T17:00:00Z",
            "homeTeam": game["home_team"], "awayTeam": game["away_team"], "markets": markets}


def test_devig_and_snapshot():
    ph, pa = nfl.devig_two_way(-298, 240)
    assert abs(ph + pa - 1) < 1e-12 and 0.70 < ph < 0.73
    assert nfl.devig_two_way(None, 240) is None and nfl.devig_two_way(float("nan"), -110) is None
    snap = nfl.market_snapshot(line=-7.0, american=-105, implied=0.4892, captured_at="t")
    assert snap == {"line": -7.0, "americanOdds": -105, "decimalOdds": 1.9524, "impliedProbability": 0.4892,
                    "source": "nflverse", "capturedAt": "t"}


def test_margin_distribution_is_anchored_and_has_key_numbers():
    assert PARAMS is not None, "sports/football/model_params.json must be committed"
    values, probs, target = nfl.home_margin_distribution(3.0, -110, -110, LADDER)
    assert abs(probs.sum() - 1) < 1e-9 and abs(target - 0.5) < 1e-12
    home = nfl.side_prices(values, probs, "home", -3.0)
    assert abs(home.win_ex_push - 0.5) < 1e-4, home          # centered on the posted line
    p3 = float(probs[values == 3][0])
    normal = mk.integer_normal_pmf(3.0, LADDER["spread_sigma"], -90, 90)[1][90 + 3]
    assert p3 > 2.0 * normal, (p3, normal)                    # key number 3 carries real mass
    assert 0.06 < home.push < 0.12, home.push                # measured push at a 3-point line: 8-12%
    assert float(probs[values == 0][0]) < 0.01               # ties are rare (0.3% of games)
    assert nfl.side_prices(values, probs, "home", -3.5).push == 0.0
    # skewed price: -120/+100 on the home side moves the anchored cover probability
    _, _, t2 = nfl.home_margin_distribution(3.0, -120, 100, LADDER)
    v2, p2, _ = nfl.home_margin_distribution(3.0, -120, 100, LADDER)
    assert abs(nfl.side_prices(v2, p2, "home", -3.0).win_ex_push - t2) < 1e-4 and t2 > 0.52


def test_pmf_pricing_agrees_with_grading():
    """Priced outcome mix == probability-weighted grade_pick over every margin."""
    values, probs, _ = nfl.home_margin_distribution(6.5, -110, -110, LADDER)
    for side in ("home", "away"):
        for line in (-13.5, -10.0, -7.0, -6.5, -3.0, 0.0, 2.5, 3.0, 7.0):
            priced = nfl.side_prices(values, probs, side, line).as_dict()
            brute = {k: 0.0 for k in mk.OUTCOMES}
            for m, p in zip(values, probs):
                brute[mk.grade_pick("spread", side, line, 20 + m, 20)[0]] += p
            for k in mk.OUTCOMES:
                assert abs(priced[k] - brute[k]) < 1e-3, (side, line, k, priced[k], brute[k])


def test_board_markets_contract():
    board = _board()
    m = board["markets"]
    ids = [x["marketId"] for x in m]
    assert len(ids) == len(set(ids)), "marketIds must be unique per board (the log grades per marketId)"
    for x in m:
        assert "edge" not in x and "confidenceTier" not in x, x            # never an edge or a tier
        assert x["basis"] in ("market", "model", "market_implied") and x["modelVersion"] == "test-v1"
        assert x["publishedAt"] and 0.0 < x["modelProbability"] < 1.0
    by = {x["marketId"].split(":", 1)[1]: x for x in m}
    ml_view = by["moneyline:full_game"]
    assert ml_view["basis"] == "model" and ml_view["side"] == "home" and ml_view["modelProbability"] == 0.68
    assert ml_view["market"]["americanOdds"] == -298 and ml_view["attribution"].startswith("Lines: nflverse")
    home, away = by["spread:full_game:home"], by["spread:full_game:away"]
    assert home["line"] == -7.0 and away["line"] == 7.0 and home["label"] == "BUF -7"   # home favoured: home line negative
    assert abs(home["modelProbability"] - home["market"]["impliedProbability"]) < 2e-4
    assert abs(home["modelProbability"] + away["modelProbability"] - 1) < 2e-4
    assert home["modelLine"] == -6.0 and away["modelLine"] == 6.0          # model fair spread: information only
    alts = sorted((x for x in m if ":spread:" in x["marketId"] and ":alt" in x["marketId"]), key=lambda x: x["line"])
    assert [a["line"] for a in alts] == [-14.0, -10.0, -4.0, 0.0] and all(a["side"] == "home" and a.get("market") is None for a in alts)
    assert all(alts[i]["modelProbability"] < alts[i + 1]["modelProbability"] for i in range(len(alts) - 1))
    over, under = by["total:full_game:over"], by["total:full_game:under"]
    assert abs(over["modelProbability"] - over["market"]["impliedProbability"]) < 2e-4 and over["line"] == 48.5
    assert abs(over["modelProbability"] + under["modelProbability"] - 1) < 1e-4
    tt = [x for x in m if x["type"].startswith("team_total_")]
    assert {x["type"] for x in tt} == {"team_total_home", "team_total_away"} and len(tt) == 4
    assert by["team_total_home:full_game:over"]["line"] == 27.5 and by["team_total_away:full_game:over"]["line"] == 20.5
    assert all(x["line"] % 1 == 0.5 and x.get("modelLine") is None for x in tt)


def test_away_favourite_and_no_line_and_stale():
    game = dict(GAME, spread_line=-3.5, home_moneyline=150, away_moneyline=-180)
    m = _board(game=game)["markets"]
    by = {x["marketId"].split(":", 1)[1]: x for x in m}
    assert by["spread:full_game:home"]["line"] == 3.5 and by["spread:full_game:away"]["line"] == -3.5
    assert all(x["side"] == "away" for x in m if ":alt" in x["marketId"] and x["type"] == "spread")
    pickem = {x["marketId"].split(":", 1)[1]: x for x in _board(game=dict(GAME, spread_line=0.0))["markets"]}
    assert json.dumps(pickem["spread:full_game:home"]["line"]) == "0.0" and pickem["spread:full_game:home"]["label"] == "BUF PK"
    no_line = {k: (None if k not in ("home_team", "away_team") else v) for k, v in GAME.items()}
    m2 = _board(game=no_line)["markets"]
    assert [x["type"] for x in m2] == ["moneyline"] and m2[0]["basis"] == "model" and "market" not in m2[0]
    assert _board(game=no_line, model=None)["markets"] == []


def test_push_gate_hides_whole_line_pushes():
    ladder = dict(LADDER, push_validated=False, total_push_validated=False)
    m = _board(ladder=ladder)["markets"]
    for x in m:
        assert "pushProbability" not in x, x
        if ":alt" in x["marketId"]:
            assert x["line"] % 1 == 0.5, x     # whole-number alt lines are not published
    assert any(x["marketId"].endswith(":spread:full_game:home") for x in m)   # the posted -7 itself still shows


def test_log_grade_and_summary_roundtrip():
    season = nfl.SEASON_OF
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        recs = ml.picks_from_boards("football", [_board()], season_of=season, model_version="test-v1",
                                    published_at="2026-10-02T07:10:00+00:00")
        ml.write_snapshot("football", recs, root=root)
        chosen = ml.pregame_picks(ml.load_snapshots("football", root=root))
        assert len(chosen) == len(recs), "every published market must survive pregame selection"
        graded = ml.grade_picks(chosen, {"2026_04_NE_BUF": (27.0, 20.0)})   # BUF by 7 (push at -7), 47 total
        assert len(graded) == len(recs) and all(g["result"] != "void" for g in graded)
        res = {g["marketId"].split(":", 1)[1]: g["result"] for g in graded}
        assert res["spread:full_game:home"] == "push" and res["spread:full_game:away"] == "push"
        assert res["moneyline:full_game"] == "win" and res["total:full_game:under"] == "win"
        assert res["team_total_home:full_game:under"] == "win" and res["team_total_away:full_game:under"] == "win"
        priced = [g for g in graded if g.get("market")]
        assert all(g["unitReturn"] is not None for g in priced)
        assert all(g["unitReturn"] is None for g in graded if not g.get("market"))   # no invented -110 ROI
        summary = {s["type"]: s for s in nfl.summarize_graded(graded)}
        assert set(summary) == {"moneyline", "spread", "alt_spread", "total", "alt_total", "team_total_home", "team_total_away"}
        assert summary["spread"]["graded"] == 1 and summary["spread"]["pushes"] == 1   # one reference side per line
        assert summary["alt_spread"]["graded"] == 4 and summary["team_total_home"]["graded"] == 1
        assert all(s["season"] == "2026-27" for s in summary.values())
        # Only the one-sided model-view moneyline is a W-L record (priced at its logged -298).
        assert nfl.RECORD_TYPES == {"moneyline"}
        mlr = summary["moneyline"]
        assert mlr["recordKind"] == "record" and mlr["wins"] == 1 and mlr["losses"] == 0
        assert abs(mlr["unitsAtStatedPrice"] - round(100 / 298, 2)) < 1e-9 and mlr["breakEvenRate"] is not None
        # Both-sides-near-50% markets and the fixed-side ladders are calibration only.
        for t in ("spread", "alt_spread", "total", "alt_total", "team_total_home", "team_total_away"):
            s = summary[t]
            assert s["recordKind"] == "calibration", t
            for k in ("wins", "losses", "winRateExPush", "breakEvenRate", "unitsAtStatedPrice", "roi"):
                assert s[k] is None, (t, k)
            assert s["hits"] is not None and s["expectedHits"] is not None, t
        over = next(g for g in graded if g["marketId"].endswith(":total:full_game:over"))
        tot = summary["total"]                                   # 47 < 48.5: the reference (over) side missed
        assert tot["hits"] == 0 and tot["expectedHits"] == round(over["modelProbability"], 2)
        assert tot["brier"] == round(over["modelProbability"] ** 2, 4) and tot["baseRateBrier"] == 0.0
        assert summary["spread"]["brier"] is None                # a push is not a binary outcome


def test_model_view_params_and_allowlist():
    assert PARAMS is not None
    forbidden = set(_LEAKY_COLUMNS) | set(mv.LINE_COLUMNS) | {"home_is_favorite", "weather_temp", "weather_wind",
                                                              "temp", "wind", "home_win", "margin", "pts_total"}
    assert "overtime" in _LEAKY_COLUMNS
    for cols in (mv.FEATURE_ALLOWLIST, mv.TOTAL_FEATURE_ALLOWLIST, PARAMS["margin_model"]["features"]):
        assert not (set(cols) & forbidden), set(cols) & forbidden
        assert not any("roster_" in c for c in cols)          # game-week statuses are post-publish
    assert PARAMS["platt"]["b"] > 0 and PARAMS["ladder"]["spread_sigma"] > 10
    frame = pd.DataFrame({c: [0.0, 5.0, -5.0] for c in PARAMS["margin_model"]["features"]})
    out = mv.predict_model_view(PARAMS, frame)
    p = out["model_home_win_probability"].to_numpy()
    assert ((p > 0) & (p < 1)).all()
    frame_nan = pd.DataFrame({"rt_margin": [np.nan]})     # every feature missing -> imputed, still a probability
    assert 0 < float(mv.predict_model_view(PARAMS, frame_nan)["model_home_win_probability"].iloc[0]) < 1


def test_committed_walk_forward_metrics_meet_acceptance():
    metrics = json.loads(mv.METRICS_PATH.read_text())
    pooled = metrics["pooled"]
    assert metrics["eval_seasons"] == [2022, 2023, 2024, 2025] and 1050 <= pooled["n"] <= 1100
    assert pooled["served"]["brier"] <= 0.2115 and pooled["served"]["log_loss"] <= 0.611
    assert pooled["model_view"]["brier"] <= 0.2250
    cal = pooled["served_calibration"]
    assert cal["slope_ci95"][0] <= 1 <= cal["slope_ci95"][1] and cal["intercept_ci95"][0] <= 0 <= cal["intercept_ci95"][1]
    lad = metrics["ladders"]
    assert lad["spread_ladder"]["analyst_offsets"]["brier"] <= 0.1960
    assert lad["total_ladder"]["analyst_offsets"]["brier"] <= 0.2025
    assert "overtime" not in metrics["feature_allowlist"]
    assert lad["push_gate_spread"]["passed"] == PARAMS["ladder"]["push_validated"]


def test_ties_are_not_labelled():
    games = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"], "season": [2024] * 3, "week": [1, 2, 3], "game_type": ["REG"] * 3,
        "gameday": ["2024-09-08", "2024-09-15", "2024-09-22"], "away_team": ["A", "A", "B"], "home_team": ["B", "B", "A"],
        "away_score": [10, 20, 3], "home_score": [10, 17, np.nan], "div_game": [0, 0, 0], "overtime": [1, 0, 0],
        "location": ["Home"] * 3, "weekday": ["Sunday"] * 3, "roof": ["outdoors"] * 3, "surface": ["grass"] * 3,
        "away_moneyline": [100, 100, 100], "home_moneyline": [-120, -120, -120], "temp": [70] * 3, "wind": [5] * 3,
    })
    g = prepare_games(games, require_completed=True)
    assert list(g["game_id"]) == ["g1", "g2"]                    # the tie is a completed game
    assert np.isnan(g.loc[g.game_id == "g1", "home_win"].iloc[0]) and g.loc[g.game_id == "g1", "is_tie"].iloc[0] == 1
    assert g.loc[g.game_id == "g2", "home_win"].iloc[0] == 0.0
    logs = build_team_game_logs(prepare_games(games, require_completed=False), pd.DataFrame())
    assert set(logs.loc[logs.game_id == "g1", "won"]) == {0.5}
    assert logs.loc[logs.game_id == "g3", "won"].isna().all()


def test_calibration_fit_on_calibrated_data():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.15, 0.85, 4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    c = mv.calibration_fit(p, y)
    assert c["slope_ci95"][0] <= 1 <= c["slope_ci95"][1] and c["intercept_ci95"][0] <= 0 <= c["intercept_ci95"][1]
    over = mv.calibration_fit(np.clip(0.5 + 1.8 * (p - 0.5), 0.01, 0.99), y)   # overconfident -> slope < 1
    assert over["slope_ci95"][1] < 1


def test_ratings_have_no_lookahead():
    rng = np.random.default_rng(5)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for week in range(1, 31):
        order = rng.permutation(teams)
        for j in range(0, 8, 2):
            rows.append({"season": 2024 + (week > 15), "week": week, "official_date": pd.Timestamp("2024-09-01") + pd.Timedelta(days=7 * week),
                         "home_team": order[j], "away_team": order[j + 1], "home_score": float(rng.integers(10, 35)),
                         "away_score": float(rng.integers(10, 35)), "is_neutral_site": 0})
    frame = pd.DataFrame(rows)
    a = mv.walkforward_ratings(frame, min_games=20)
    changed = frame.copy()
    changed.loc[changed["week"] >= 25, ["home_score", "away_score"]] = [60.0, 0.0]   # rewrite the future
    b = mv.walkforward_ratings(changed, min_games=20)
    early = frame["week"] < 25
    assert a.loc[early, "rt_margin"].notna().any()
    assert np.allclose(a.loc[early, "rt_margin"].fillna(0), b.loc[early, "rt_margin"].fillna(0))
    assert not np.allclose(a.loc[frame["week"] >= 26, "rt_margin"], b.loc[frame["week"] >= 26, "rt_margin"])


def test_history_prob_is_null_on_market_basis_rows():
    """Shipped apps render history `prob` as "Model confidence": never the sportsbook price."""
    import scripts.export_football_frontend_data as ex
    frame = pd.DataFrame({
        "game_id": ["g_mkt", "g_model"], "season": [2025, 2025], "official_date": pd.to_datetime(["2025-10-05", "2025-10-12"]),
        "home_team": ["BUF", "KC"], "away_team": ["NE", "LV"], "home_score": [27.0, 17.0], "away_score": [20.0, 24.0],
        "home_win": [1.0, 0.0], "home_win_probability": [0.74, 0.61], "model_home_win_probability": [0.69, 0.61],
        "headline_basis": ["market", "model"],
    })
    player_week = pd.DataFrame(columns=["game_id", "team", "position", "player_id", "fantasy_points", "targets", "carries"])
    saved = ex._history_frame, ex._load_table_optional
    try:
        ex._history_frame = lambda games_df, params, seasons: frame.copy()
        ex._load_table_optional = lambda stem: player_week.copy()
        boards = {b["gameId"]: b for b in ex._historical_boards(games_df=frame, params={})}
    finally:
        ex._history_frame, ex._load_table_optional = saved
    mkt, model = boards["g_mkt"], boards["g_model"]
    assert mkt["basis"] == "market" and mkt["prob"] is None
    assert mkt["homeWinProbability"] == 0.74 and mkt["predictedWinner"] == "BUF" and mkt["modelHomeWinProbability"] == 0.69
    assert mkt["predictionSource"] == "market_devig_nflverse_close" and mkt["attribution"].startswith("Lines: nflverse")
    assert model["basis"] == "model" and model["prob"] == 0.61 and model["attribution"] is None
    assert all(b["recordBasis"] == "simulated" for b in boards.values())


def test_game_start_utc():
    from scripts.export_football_frontend_data import _game_start_utc
    assert _game_start_utc("2026-10-04", "13:00") == "2026-10-04T17:00:00Z"     # EDT
    assert _game_start_utc("2026-12-06", "13:00") == "2026-12-06T18:00:00Z"     # EST
    assert _game_start_utc("2026-10-11", "09:30") == "2026-10-11T13:30:00Z"     # London game, listed in ET
    assert _game_start_utc(None, "13:00") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_football_markets: all checks passed")
