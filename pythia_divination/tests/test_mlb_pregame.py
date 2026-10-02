"""Checks for the honest pregame MLB model, its markets and the post-game leak fixes.

Run (cwd = pythia_divination):  python tests/test_mlb_pregame.py

No network and no data files: every check runs on synthetic games/payloads.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATABASE_URL", "postgresql://unused:unused@127.0.0.1:5432/none")

from sports import market_log as ml  # noqa: E402
from sports import markets as mk  # noqa: E402
from sports.mlb import pregame_model as pm  # noqa: E402
from sports.mlb.feature_engineering import _parse_wind  # noqa: E402
from sports.mlb.markets_mlb import fair_half_line, mlb_market_picks  # noqa: E402
from sports.mlb.roster_features import (  # noqa: E402
    build_pregame_bullpen_roster,
    extract_roster_tables,
    starting_lineup,
)

PARAMS = pm.load_params()


# ── synthetic season ─────────────────────────────────────────────────────────

def _synthetic_games(n_days: int = 150, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six teams, one venue each, a starter rotation per team, scores driven by a hidden
    team strength. Spans two seasons so season regression and month refits happen."""
    rng = np.random.default_rng(seed)
    teams = list(range(101, 107))
    strength = dict(zip(teams, rng.normal(0, 0.35, len(teams))))
    rows, lines, pk = [], [], 1000
    days = list(pd.date_range("2024-04-01", periods=n_days // 2)) + list(pd.date_range("2025-04-01", periods=n_days // 2))
    for day in days:
        order = rng.permutation(teams)
        for h, a in ((order[0], order[1]), (order[2], order[3]), (order[4], order[5])):
            mu_h = 4.6 * math.exp(0.25 * (strength[h] - strength[a]) + 0.03)
            mu_a = 4.4 * math.exp(0.25 * (strength[a] - strength[h]))
            hs, as_ = int(rng.poisson(mu_h)), int(rng.poisson(mu_a))
            if hs == as_:
                hs += 1
            pk += 1
            sp_h, sp_a = 1000 * h + pk % 5, 1000 * a + (pk + 2) % 5
            rows.append({"game_pk": pk, "game_type": "R", "season": day.year, "game_date": (day + pd.Timedelta(hours=23)).isoformat(),
                         "official_date": day.date().isoformat(), "game_number": 1, "status_abstract": "Final",
                         "status_detailed": "Final", "away_team_id": int(a), "away_team_name": f"T{a}",
                         "home_team_id": int(h), "home_team_name": f"T{h}", "away_score": as_, "home_score": hs,
                         "away_probable_pitcher_id": sp_a, "away_probable_pitcher_name": f"P{sp_a}",
                         "home_probable_pitcher_id": sp_h, "home_probable_pitcher_name": f"P{sp_h}",
                         "venue_id": int(h), "venue_name": f"Park{h}", "day_night": "night", "double_header": "N",
                         "scheduled_innings": 9, "series_description": "Regular Season", "weather_condition": "Clear",
                         "weather_temp_f": float(rng.integers(55, 95)), "weather_wind": f"{int(rng.integers(0, 15))} mph, Out To CF"})
            for team, sp, allowed in ((h, sp_h, as_), (a, sp_a, hs)):
                ip = float(rng.integers(12, 21)) / 3.0
                lines.append({"game_pk": pk, "team_id": int(team), "starter_id": sp, "ip": ip, "er": min(allowed, 5),
                              "bb": int(rng.integers(0, 4)), "so": int(rng.integers(2, 10)), "hr": int(rng.integers(0, 3)),
                              "bf": int(ip * 4.3), "h": int(rng.integers(2, 9)), "pitches": 90, "strikes": 58})
    # three scheduled (upcoming) games after the last completed day
    last = days[-1]
    for k, (h, a) in enumerate(((101, 102), (103, 104), (105, 106))):
        pk += 1
        rows.append({**rows[-1], "game_pk": pk, "official_date": (last + pd.Timedelta(days=1)).date().isoformat(),
                     "game_date": (last + pd.Timedelta(days=1, hours=23)).isoformat(), "status_abstract": "Preview",
                     "status_detailed": "Scheduled", "home_team_id": h, "away_team_id": a, "home_score": None,
                     "away_score": None, "home_probable_pitcher_id": 1000 * h + 1, "away_probable_pitcher_id": 1000 * a + 1,
                     "venue_id": h, "weather_temp_f": None, "weather_wind": None, "weather_condition": None})
    games = pm._normalize_results(pd.DataFrame(rows))
    return games, pd.DataFrame(lines)


# ── tests ────────────────────────────────────────────────────────────────────

def test_wind_regex_fixed():
    assert _parse_wind("14 mph, Out To CF") == (14.0, "Out To CF")
    assert _parse_wind("0 mph, None") == (0.0, "None")
    assert pm.wind_out_mph("19 mph, In From RF", "Overcast") == -19.0
    assert pm.wind_out_mph("8 mph, Out To CF", "Partly Cloudy") == 8.0
    assert pm.wind_out_mph("8 mph, L To R", "Sunny") == 0.0
    assert pm.wind_out_mph("10 mph, Out To CF", "Roof Closed") == 0.0


def test_starting_lineup_is_true_starters_not_end_of_game_order():
    players = {f"ID{100 + s}": {"person": {"id": 100 + s}, "battingOrder": f"{s}00"} for s in range(1, 10)}
    players["ID999"] = {"person": {"id": 999}, "battingOrder": "301"}   # pinch hitter for slot 3
    players["ID555"] = {"person": {"id": 555}}                          # reliever, never batted
    team_box = {"players": players, "battingOrder": [101, 102, 999, 104, 105, 106, 107, 108, 109]}
    assert starting_lineup(team_box) == [(s, 100 + s) for s in range(1, 10)]


def test_pregame_bullpen_only_prior_appearances():
    logs = pd.DataFrame({"team_id": [1, 1, 1, 1, 2], "reliever_id": [11, 12, 13, 14, 21],
                         "official_date": ["2025-05-01", "2025-05-10", "2025-05-20", "2025-05-30", "2025-05-19"]})
    games = pd.DataFrame({"game_pk": [7], "official_date": ["2025-05-20"], "season": [2025], "team_side": ["home"],
                          "team_id": [1], "team_name": ["A"]})
    roster = build_pregame_bullpen_roster(logs, games, lookback_days=14)
    # 11 is >14 days old, 13 pitched on the game date itself, 14 later, 21 for another team
    assert roster["reliever_id"].tolist() == [12], roster


def test_extract_roster_tables_never_reads_unused_bullpen_or_end_order():
    def payload(game_pk: int, date: str, used_reliever: int) -> dict:
        teams = {}
        for side, tid in (("home", 1), ("away", 2)):
            players = {f"ID{tid * 100 + s}": {"person": {"id": tid * 100 + s}, "battingOrder": f"{s}00",
                                              "stats": {"batting": {"plateAppearances": 4, "atBats": 4, "hits": 1}}}
                       for s in range(1, 10)}
            players[f"ID{tid * 1000}"] = {"person": {"id": tid * 1000}, "battingOrder": "201",
                                          "stats": {"batting": {"plateAppearances": 1, "atBats": 1}}}
            players[f"ID{used_reliever + tid}"] = {"person": {"id": used_reliever + tid},
                                                   "stats": {"pitching": {"gamesStarted": 0, "inningsPitched": "0.0",
                                                                          "battersFaced": 3, "earnedRuns": 3}}}
            teams[side] = {"team": {"id": tid, "name": f"T{tid}"}, "players": players,
                           "battingOrder": [tid * 100 + 1, tid * 1000] + [tid * 100 + s for s in range(3, 10)],
                           "bullpen": [777, 778, 779]}  # relievers who did NOT pitch: must be ignored
        return {"game_pk": game_pk, "live_feed": {"gamePk": game_pk, "gameData": {
            "game": {"season": "2025"}, "datetime": {"officialDate": date},
            "teams": {"home": {"id": 1, "name": "T1"}, "away": {"id": 2, "name": "T2"}}, "players": {}}},
            "boxscore": {"teams": teams}}

    with tempfile.TemporaryDirectory() as d:
        Path(d, "game_1.json").write_text(json.dumps(payload(1, "2025-05-01", 500)))
        Path(d, "game_2.json").write_text(json.dumps(payload(2, "2025-05-03", 600)))
        lineup, batters, bullpen, relievers = extract_roster_tables([Path(d)])
    assert set(lineup["batter_id"]) == {100 + s for s in range(1, 10)} | {200 + s for s in range(1, 10)}
    assert not ({777, 778, 779} & set(bullpen["reliever_id"])), "unused-bullpen list leaked into the roster"
    # game 1 has no earlier appearances; game 2's bullpen is game 1's (0-out!) relievers only
    assert bullpen.loc[bullpen["game_pk"] == 1].empty
    assert sorted(bullpen.loc[bullpen["game_pk"] == 2, "reliever_id"]) == [501, 502]
    assert len(relievers) == 4  # 0-out outings are kept


def test_features_are_pregame():
    """Changing a game's result must not change that game's own features (or any earlier
    game's), only later games'."""
    games, lines = _synthetic_games(80)
    base = pm.build_features(games, lines, PARAMS)
    target = base[(base["season"] == 2025) & base["is_final"]].iloc[20]
    altered = games.copy()
    m = altered["game_pk"] == target["game_pk"]
    altered.loc[m, ["home_score", "away_score"]] = [0, 15]
    altered.loc[m, "home_win"] = 0.0
    lines2 = lines.copy()
    lines2.loc[lines2["game_pk"] == target["game_pk"], ["er", "hr", "bb"]] = [9, 4, 6]
    alt = pm.build_features(altered, lines2, PARAMS)
    feats = sorted(set(PARAMS["win_model"]["features"]) | {"h_rs_pg", "a_rs_pg", "h_ra_pg", "a_ra_pg", "elo_diff"})
    a = base.set_index("game_pk")[feats]; b = alt.set_index("game_pk")[feats]
    upto = base.loc[base["order"] <= target["order"], "game_pk"]
    assert np.allclose(a.loc[upto].to_numpy(float), b.loc[upto].to_numpy(float), equal_nan=True), "same-game leak"
    later = base.loc[base["order"] > target["order"], "game_pk"]
    assert not np.allclose(a.loc[later].to_numpy(float), b.loc[later].to_numpy(float), equal_nan=True)


def test_walk_forward_predictions_ignore_later_results():
    games, lines = _synthetic_games(120)
    F = pm.build_features(games, lines, PARAMS)
    start = pd.Timestamp("2025-04-01")
    pred, _ = pm.walk_forward(F, PARAMS, start=start)
    cut = pd.Timestamp("2025-05-10")
    flipped = games.copy()
    m = flipped["official_date"] >= cut
    flipped.loc[m, ["home_score", "away_score"]] = flipped.loc[m, ["away_score", "home_score"]].to_numpy()
    flipped["home_win"] = np.where(flipped["is_final"], (flipped["home_score"] > flipped["away_score"]).astype(float), np.nan)
    pred2, _ = pm.walk_forward(pm.build_features(flipped, lines, PARAMS), PARAMS, start=start)
    before = F.loc[F["official_date"] < cut, "game_pk"]
    p1 = pred[pred["game_pk"].isin(before)].set_index("game_pk")
    assert len(p1) > 50
    p2 = pred2.set_index("game_pk").loc[p1.index]
    assert np.allclose(p1[["p_home", "mu_home", "mu_away"]].to_numpy(float), p2[["p_home", "mu_home", "mu_away"]].to_numpy(float))
    # every prediction's model was fit strictly before the game's month
    merged = pred.merge(F[["game_pk", "official_date"]], on="game_pk")
    assert (pd.to_datetime(merged["train_cutoff"]) <= merged["official_date"]).all()
    assert (pd.to_datetime(merged["train_cutoff"]) == merged["official_date"].dt.to_period("M").dt.to_timestamp()).all()
    # upcoming (unplayed) games get a prediction too, from the same procedure
    upcoming = F.loc[~F["is_final"], "game_pk"]
    assert pred["game_pk"].isin(upcoming).sum() == len(upcoming)
    assert pred["p_home"].between(0.05, 0.95).all() and pred["nb_r"].gt(0).all()


def test_score_matrix_and_fair_lines():
    m = pm.score_matrix(4.6, 4.1, 3.7)
    assert abs(m.sum() - 1.0) < 1e-9
    i, j = np.indices(m.shape)
    assert abs((m.sum(axis=1) * np.arange(m.shape[0])).sum() - 4.6) < 0.01
    total = np.bincount((i + j).ravel(), weights=m.ravel())
    line = fair_half_line(total)
    assert line % 1 == 0.5
    p_over = total[np.arange(len(total)) > line].sum()
    for other in (line - 1, line + 1):
        assert abs(p_over - 0.5) <= abs(total[np.arange(len(total)) > other].sum() - 0.5) + 1e-12


def test_market_entries_contract_and_grading_agreement():
    entries = mlb_market_picks(board_id="mlb-1", home_label="NYY", away_label="BOS", p_home=0.56, mu_home=4.9,
                               mu_away=4.2, nb_r=3.7, published_at="2026-10-02T07:10:00+00:00",
                               model_version="mlb-pregame-v1@2026-10-01")
    assert [e["type"] for e in entries] == ["run_line", "total", "team_total_home", "team_total_away"]
    assert len({e["marketId"] for e in entries}) == 4
    for e in entries:
        assert e["basis"] == "model" and e["modelVersion"].startswith("mlb-pregame")
        assert "edge" not in e and "market" not in e
        assert e["confidenceTier"] in ("low", "medium", "high")
        assert abs(sum(e["outcomeProbabilities"].values()) - 1.0) < 1e-3
        assert e["line"] % 1 == 0.5 or e["line"] == -1.5
    rl = entries[0]
    assert rl["side"] == "home" and rl["label"] == "NYY -1.5" and rl["line"] == -1.5
    # priced probabilities == probability-weighted grade_pick outcomes over the score matrix
    m = pm.score_matrix(4.9, 4.2, 3.7)
    for e in entries:
        win = sum(m[h, a] for h in range(m.shape[0]) for a in range(m.shape[1])
                  if mk.grade_pick(e["type"], e["side"], e["line"], h, a)[0] == "win")
        assert abs(win - e["outcomeProbabilities"]["win"]) < 1e-3, (e["type"], win, e["outcomeProbabilities"])
    away_fav = mlb_market_picks(board_id="mlb-2", home_label="NYY", away_label="BOS", p_home=0.45, mu_home=4.0,
                                mu_away=4.8, nb_r=3.7, published_at=None, model_version="v")
    assert away_fav[0]["side"] == "away" and away_fav[0]["label"] == "BOS -1.5"
    assert mlb_market_picks(board_id="mlb-3", home_label="A", away_label="B", p_home=0.5, mu_home=float("nan"),
                            mu_away=4.0, nb_r=3.7, published_at=None, model_version="v") == []


def test_market_log_roundtrip_unpriced():
    board = {"id": "mlb-9", "gameId": "9", "scheduledDate": 20261003, "gameStart": "2026-10-03T17:00:00Z",
             "homeTeam": "Cleveland Guardians", "awayTeam": "Chicago White Sox",
             "markets": mlb_market_picks(board_id="mlb-9", home_label="CLE", away_label="CWS", p_home=0.58,
                                         mu_home=4.3, mu_away=3.9, nb_r=3.65, published_at=None, model_version="v1")}
    with tempfile.TemporaryDirectory() as d:
        recs = ml.picks_from_boards("mlb", [board], season_of=ml.season_calendar, model_version="v1",
                                    published_at="2026-10-02T07:10:00+00:00")
        late = ml.picks_from_boards("mlb", [board], season_of=ml.season_calendar, model_version="v1",
                                    published_at="2026-10-03T18:00:00+00:00")  # after first pitch: never graded
        ml.write_snapshot("mlb", recs, root=Path(d))
        ml.write_snapshot("mlb", late, root=Path(d))
        chosen = ml.pregame_picks(ml.load_snapshots("mlb", root=Path(d)))
        assert len(chosen) == 4 and all(p["publishedAt"].startswith("2026-10-02") for p in chosen.values())
        graded = ml.grade_picks(chosen, {"9": (5.0, 2.0)})
    by_type = {g["type"]: g for g in graded}
    assert by_type["run_line"]["result"] == "win"            # CLE -1.5, won by 3
    assert by_type["total"]["result"] in ("win", "loss")
    assert all(g["unitReturn"] is None for g in graded)      # no line feed -> no invented ROI
    assert all(g["season"] == "2026" for g in graded)
    summary = ml.summarize(graded)
    assert all(s["roi"] is None and s["unitsAtStatedPrice"] is None for s in summary)


def test_history_board_semantics():
    from scripts.export_mlb_frontend_data import _history_board

    row = pd.Series({"game_pk": 5, "home_win_probability": 0.4999, "home_win": 0.0, "home_team_name": "H",
                     "away_team_name": "A", "season": 2026, "official_date": pd.Timestamp("2026-09-01"),
                     "venue_name": "Park", "home_score": 1.0, "away_score": 3.0, "home_probable_pitcher_name": "x",
                     "away_probable_pitcher_name": "y", "model_version": "v@2026-09-01", "home_team_id": 147,
                     "away_team_id": 111})
    board = _history_board(row, detailed=False, lineups={})
    assert board["predictedWinner"] == "A" and board["hitStatus"] == "Top Pick"
    assert board["predictions"][0]["side"] == "away" and board["predictions"][0]["rank"] == 1
    assert board["tournamentId"] == "mlb-5" and board["gameId"] == "5"
    for key in ("year", "tournament", "tour", "predictedTop3", "predictedTop5", "actualWinner", "prob", "fullField",
                "latestDate", "scheduledDate", "awayTeamDetails", "homeTeamDetails", "awayLineup", "homeLineup"):
        assert key in board, key
    row2 = row.copy(); row2["home_win_probability"] = 0.62
    assert _history_board(row2, detailed=False, lineups={})["hitStatus"] == "Miss"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
