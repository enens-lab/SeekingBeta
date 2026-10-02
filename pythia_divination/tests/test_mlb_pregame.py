"""Checks for the honest pregame MLB model, its markets and the post-game leak fixes.

Run (cwd = pythia_divination):  python tests/test_mlb_pregame.py

No network and no data files: every check runs on synthetic games/payloads.
"""
from __future__ import annotations

import contextlib
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
    assert [e["type"] for e in entries] == ["run_line", "run_line", "total", "team_total_home", "team_total_away"]
    assert len({e["marketId"] for e in entries}) == 5
    for e in entries:
        assert e["basis"] == "model" and e["modelVersion"].startswith("mlb-pregame")
        assert "edge" not in e and "market" not in e
        assert "confidenceTier" not in e  # no calibrated MLB tier cut points: never a tier
        assert abs(sum(e["outcomeProbabilities"].values()) - 1.0) < 1e-3
        assert e["line"] % 1 == 0.5 or e["line"] == -1.5
        assert "pushProbability" not in e  # x.5 lines never push
    fav, dog = entries[0], entries[1]
    # home -1.5 (its loss = away +1.5) and away -1.5, the model favourite's first
    assert (fav["side"], fav["label"], fav["line"]) == ("home", "NYY -1.5", -1.5)
    assert (dog["side"], dog["label"], dog["line"]) == ("away", "BOS -1.5", -1.5)
    assert fav["marketId"] == "mlb-1:run_line:full_game:home" and dog["marketId"] == "mlb-1:run_line:full_game:away"
    assert fav["modelProbability"] > dog["modelProbability"]
    assert fav["modelLine"] < 0 < dog["modelLine"] and abs(fav["modelLine"] + dog["modelLine"]) < 1e-9
    # priced probabilities == probability-weighted grade_pick outcomes over the score matrix
    m = pm.score_matrix(4.9, 4.2, 3.7)
    i, j = np.indices(m.shape)
    for e in entries:
        win = sum(m[h, a] for h in range(m.shape[0]) for a in range(m.shape[1])
                  if mk.grade_pick(e["type"], e["side"], e["line"], h, a)[0] == "win")
        assert abs(win - e["outcomeProbabilities"]["win"]) < 1e-3, (e["type"], win, e["outcomeProbabilities"])
    assert abs(fav["outcomeProbabilities"]["loss"] - m[i - j < 2].sum()) < 1e-3  # = away +1.5 covers
    away_fav = mlb_market_picks(board_id="mlb-1", home_label="NYY", away_label="BOS", p_home=0.45, mu_home=4.0,
                                mu_away=4.8, nb_r=3.7, published_at=None, model_version="v")
    assert away_fav[0]["side"] == "away" and away_fav[0]["label"] == "BOS -1.5"
    # a favourite flipping between bakes keeps each side's marketId (no withdrawn picks)
    assert {e["marketId"] for e in away_fav[:2]} == {fav["marketId"], dog["marketId"]}
    assert mlb_market_picks(board_id="mlb-3", home_label="A", away_label="B", p_home=0.5, mu_home=float("nan"),
                            mu_away=4.0, nb_r=3.7, published_at=None, model_version="v") == []


def test_batch_probabilities_match_single_matrix():
    mu_h, mu_a, rs = [4.9, 3.6, 5.4], [4.2, 4.4, 3.1], [3.7, 3.7, 3.2]
    pr = pm.batch_probabilities(mu_h, mu_a, rs, total_lines=(8.5,), team_lines=(4.5,))
    for k, (h, a, r) in enumerate(zip(mu_h, mu_a, rs)):
        m = pm.score_matrix(h, a, r)
        i, j = np.indices(m.shape)
        assert abs(pr["home_by_2"][k] - m[i - j >= 2].sum()) < 1e-12
        assert abs(pr["away_by_2"][k] - m[j - i >= 2].sum()) < 1e-12
        assert abs(pr["over_8.5"][k] - m[(i + j) > 8.5].sum()) < 1e-12
        assert abs(pr["home_over_4.5"][k] - m.sum(axis=1)[5:].sum()) < 1e-12
        total = np.bincount((i + j).ravel(), weights=m.ravel())
        assert pr["total_fair_line"][k] == fair_half_line(total)
        assert pr["home_fair_line"][k] == fair_half_line(m.sum(axis=1))
        entries = mlb_market_picks(board_id="b", home_label="H", away_label="A", p_home=0.5, mu_home=h, mu_away=a,
                                   nb_r=r, published_at=None, model_version="v")
        served_total = next(e for e in entries if e["type"] == "total")
        assert served_total["line"] == pr["total_fair_line"][k]
        assert abs(served_total["modelProbability"] - pr["over_fair"][k]) < 1e-4


def test_completed_states_and_result_status():
    assert pm.is_completed_state("Final") and pm.is_completed_state("Completed Early: Rain")
    assert pm.is_completed_state("Game Over") and not pm.is_completed_state("Postponed")
    assert not pm.is_completed_state("In Progress") and not pm.is_completed_state(None)
    assert pm.result_status("Completed Early: Rain") == "shortened" and pm.result_status("Final") == "final"
    games, _ = _synthetic_games(20)
    raw = games.drop(columns=["dt", "is_final", "home_win", "is_postseason", "order"]).copy()
    raw.loc[raw.index[0], "status_detailed"] = "Completed Early: Rain"
    raw.loc[raw.index[1], ["status_detailed", "home_score", "away_score"]] = ["Postponed", None, None]
    norm = pm._normalize_results(raw)
    assert bool(norm.set_index("game_pk").loc[raw.iloc[0]["game_pk"], "is_final"])
    assert raw.iloc[1]["game_pk"] not in set(norm["game_pk"])


def _market_engine(rows: list[dict], *, listings: list[dict] | None = None, live_seasons=()):
    """Engine stub: `rows` is engine.frame; `listings` the raw live schedule rows
    (game_pk, status_detailed, game_date, home_score, away_score) of `live_seasons`."""
    from scripts.export_mlb_frontend_data import PregameEngine
    frame = pd.DataFrame(rows)
    frame["dt"] = pd.to_datetime(frame["dt"])
    sched = pd.DataFrame(listings or [], columns=pm.LIVE_SCHEDULE_COLUMNS + ["fetched_season"])
    return PregameEngine(frame=frame, params={"version": "mlb-pregame-v1"}, built_at=0.0,
                         live_schedule=sched, live_seasons=frozenset(live_seasons))


def _listing(pk: int, state: str, when: str, home=None, away=None, season: int = 2026) -> dict:
    return {"game_pk": pk, "season": season, "game_type": "R", "official_date": when[:10], "game_date": when,
            "status_abstract": "Final" if state != "Scheduled" else "Preview", "status_detailed": state,
            "home_score": home, "away_score": away, "fetched_season": season}


def _market_records(pks, *, start: str = "2026-09-20T23:05:00Z", date_key: int = 20260920) -> list[dict]:
    boards = [{"id": f"mlb-{pk}", "gameId": str(pk), "scheduledDate": date_key, "gameStart": start,
               "homeTeam": "H", "awayTeam": "A",
               "markets": mlb_market_picks(board_id=f"mlb-{pk}", home_label="H", away_label="A", p_home=0.55,
                                           mu_home=4.6, mu_away=4.1, nb_r=3.7,
                                           published_at="2026-09-20T07:10:00+00:00", model_version="v")}
              for pk in pks]
    return ml.picks_from_boards("mlb", boards, season_of=ml.season_calendar, model_version="v")


@contextlib.contextmanager
def _patched(*patches):
    """(obj, attr, value) monkeypatches, restored on exit (keeps tests zero-arg so the
    __main__ runner and pytest both work)."""
    saved = []
    try:
        for obj, name, value in patches:
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def test_market_results_void_rules_and_grading():
    import scripts.export_mlb_frontend_data as ex

    start = "2026-09-20T23:05:00Z"
    engine = _market_engine([
        # played as listed
        {"game_pk": 1, "is_final": True, "home_score": 5.0, "away_score": 2.0, "dt": "2026-09-20 23:05", "status_detailed": "Final"},
        # rain-shortened official game: moneyline stands, run line / totals void
        {"game_pk": 2, "is_final": True, "home_score": 4.0, "away_score": 1.0, "dt": "2026-09-20 23:05", "status_detailed": "Completed Early: Rain"},
        # postponed, made up two days later under the same gamePk
        {"game_pk": 3, "is_final": True, "home_score": 6.0, "away_score": 0.0, "dt": "2026-09-22 17:10", "status_detailed": "Final"},
        # still to be played (after `now`)
        {"game_pk": 5, "is_final": False, "home_score": None, "away_score": None, "dt": "2026-09-24 23:05", "status_detailed": "Scheduled"},
    ], listings=[
        _listing(1, "Final", "2026-09-20T23:05:00Z", 5, 2),
        _listing(2, "Completed Early: Rain", "2026-09-20T23:05:00Z", 4, 1),
        _listing(3, "Postponed", "2026-09-20T23:05:00Z"),
        _listing(3, "Final", "2026-09-22T17:10:00Z", 6, 0),
        # game 4: rained out, no make-up date yet (the engine frame drops it)
        _listing(4, "Postponed", "2026-09-20T23:05:00Z"),
        _listing(5, "Scheduled", "2026-09-24T23:05:00Z"),
    ], live_seasons={2026})
    boards = []
    for pk in (1, 2, 3, 4, 5):
        boards.append({"id": f"mlb-{pk}", "gameId": str(pk), "scheduledDate": 20260920 if pk != 5 else 20260924,
                       "gameStart": start if pk != 5 else "2026-09-24T23:05:00Z", "homeTeam": "H", "awayTeam": "A",
                       "markets": mlb_market_picks(board_id=f"mlb-{pk}", home_label="H", away_label="A", p_home=0.55,
                                                   mu_home=4.6, mu_away=4.1, nb_r=3.7,
                                                   published_at="2026-09-20T07:10:00+00:00", model_version="v")})
    records = ml.picks_from_boards("mlb", boards, season_of=ml.season_calendar, model_version="v")
    now = pd.Timestamp("2026-09-23T12:00:00Z")
    results = ex._market_results(engine, records, now=now)
    assert results["1"] == {"home": 5.0, "away": 2.0, "status": "final"}
    assert results["2"]["status"] == "shortened"
    assert results["3"]["status"] == "postponed"          # made up > 24 h after the listed start
    assert results["4"] == {"home": None, "away": None, "status": "postponed"}  # live "Postponed", never played
    assert "5" not in results                               # not started yet: stays ungraded
    # before the 24 h window passes an unplayed game is not voided yet
    assert "4" not in ex._market_results(engine, records, now=pd.Timestamp("2026-09-21T12:00:00Z"))

    graded = ex._grade_market_picks(ml.pregame_picks(records), results)
    assert all(g["basis"] == "model" for g in graded)
    by_game: dict[str, list] = {}
    for g in graded:
        by_game.setdefault(g["gameId"], []).append(g)
    assert {g["result"] for g in by_game["1"]} <= {"win", "loss"} and len(by_game["1"]) == 5
    home_rl = next(g for g in by_game["1"] if g["marketId"].endswith(":run_line:full_game:home"))
    assert home_rl["result"] == "win" and home_rl["unitReturn"] is None   # won by 3, unpriced
    assert "5" not in by_game
    if ex._GRADER_TAKES_STATUS:
        # shared grader with status support: void, never a win/loss on the wrong game
        assert all(g["result"] == "void" for g in by_game["2"] + by_game["3"] + by_game["4"])
    else:
        # older grader (scores only): such games are left ungraded rather than mis-graded
        assert not ({"2", "3", "4"} & set(by_game))
    summary = ml.summarize(graded)
    assert all(s["roi"] is None for s in summary)
    # every MLB type is calibration-only (fixed-side run lines, fair-line totals): no W-L
    assert summary and all(s["recordKind"] == "calibration" for s in summary)
    assert all(s["wins"] is None and s["losses"] is None and s["winRateExPush"] is None for s in summary)
    assert all(s["hits"] is not None and s["expectedHits"] is not None for s in summary)


def test_market_results_need_positive_evidence_to_void():
    import scripts.export_mlb_frontend_data as ex

    now = pd.Timestamp("2026-09-23T12:00:00Z")   # > 24 h after every listed start
    records = _market_records((11, 12, 13, 14, 15, 16))
    engine = _market_engine(
        # the engine frame knows none of these games (e.g. outside its prediction window)
        [{"game_pk": 99, "is_final": True, "home_score": 1.0, "away_score": 0.0, "dt": "2026-09-19 23:05",
          "status_detailed": "Final"}],
        listings=[
            _listing(11, "Final", "2026-09-20T23:05:00Z", 3, 7),      # final in the live schedule
            _listing(12, "Postponed", "2026-09-20T23:05:00Z"),         # live postponement
            _listing(13, "Cancelled", "2026-09-20T23:05:00Z"),
            _listing(14, "Suspended: Rain", "2026-09-20T23:05:00Z"),
            # 15: absent from the live 2026 schedule (an unneeded "if necessary" game)
            _listing(16, "Postponed", "2026-09-20T23:05:00Z"),         # ... made up inside the window,
            _listing(16, "In Progress", "2026-09-21T20:05:00Z"),      # not final yet: wait
        ], live_seasons={2026})
    results = ex._market_results(engine, records, now=now)
    assert results["11"] == {"home": 3.0, "away": 7.0, "status": "final"}   # graded, never voided
    assert results["12"]["status"] == "postponed" and results["12"]["home"] is None
    assert results["13"]["status"] == "cancelled"
    assert results["14"]["status"] == "postponed"
    assert results["15"]["status"] == "postponed"
    assert "16" not in results
    graded = ex._grade_market_picks(ml.pregame_picks(records), results)
    by_game: dict[str, set] = {}
    for g in graded:
        by_game.setdefault(g["gameId"], set()).add(g["result"])
    assert by_game["11"] <= {"win", "loss"}
    assert all(by_game[gid] == {"void"} for gid in ("12", "13", "14", "15"))
    # inside the 24 h window nothing is voided yet, even with a live "Postponed"
    early = ex._market_results(engine, records, now=pd.Timestamp("2026-09-21T12:00:00Z"))
    assert set(early) == {"11"}


def test_market_results_fetch_failure_voids_nothing():
    """The verifier's case: the live schedule was not fetched (failure / local fallback),
    so every logged game is missing from engine.frame. None may be voided."""
    import scripts.export_mlb_frontend_data as ex

    records = _market_records(range(100, 140))
    engine = _market_engine([{"game_pk": 1, "is_final": True, "home_score": 1.0, "away_score": 0.0,
                              "dt": "2026-07-20 23:05", "status_detailed": "Final"}])   # no listings, no live seasons
    assert ex._market_results(engine, records, now=pd.Timestamp("2026-10-01T12:00:00Z")) == {}
    # a live schedule fetched for ANOTHER season is no evidence about 2026 games either
    engine_2025 = _market_engine([{"game_pk": 1, "is_final": True, "home_score": 1.0, "away_score": 0.0,
                                   "dt": "2026-07-20 23:05", "status_detailed": "Final"}],
                                 listings=[_listing(7, "Final", "2025-07-01T23:05:00Z", 1, 0, season=2025)],
                                 live_seasons={2025})
    assert ex._market_results(engine_2025, records, now=pd.Timestamp("2026-10-01T12:00:00Z")) == {}


def _raw_results(seasons) -> pd.DataFrame:
    """Synthetic games in flatten_results shape for the given seasons."""
    games, _ = _synthetic_games(40)
    raw = games.drop(columns=["dt", "is_final", "home_win", "is_postseason", "order"])
    raw["official_date"] = raw["official_date"].dt.strftime("%Y-%m-%d")
    return raw[raw["season"].isin(list(seasons))].reset_index(drop=True)


def test_load_results_requires_live_for_required_seasons():
    import requests

    calls: list[int] = []

    def fake_fetch(client, season, **_):
        calls.append(int(season))
        if season in (2021, 2026):
            raise requests.Timeout("read timed out")
        return _raw_results([season]) if season in (2024, 2025) else pd.DataFrame()

    def fake_local(season, normalized_dir=None):
        if season == 2026:   # the stale S3 table the old code silently fell back to
            stale = _raw_results([2025])
            return stale.assign(season=2026, game_pk=stale["game_pk"] + 100_000)
        raise FileNotFoundError(str(season))

    with _patched((pm, "fetch_season_results", fake_fetch), (pm, "load_local_season", fake_local),
                  (pm, "LIVE_FETCH_BACKOFF_SECONDS", 0.0)):
        try:
            pm.load_results(range(2020, 2027), client=object(), require_live=[2024, 2025, 2026])
        except pm.LiveDataError as exc:
            assert "2026" in str(exc)
        else:
            raise AssertionError("a failed current-season fetch must raise, not fall back to local tables")
        # retried, and the failing training-only season (2021) fell back without raising
        assert calls.count(2026) == pm.LIVE_FETCH_ATTEMPTS and calls.count(2021) == pm.LIVE_FETCH_ATTEMPTS
        # not required -> the old fallback (used by the offline fit script) still works
        sink: list = []
        games = pm.load_results(range(2024, 2027), client=object(), schedule_sink=sink)
        assert set(games["season"]) == {2024, 2025, 2026}
        assert {int(s) for f in sink for s in f["fetched_season"]} == {2024, 2025}   # only live seasons
        # required + an EMPTY live payload while the local table has games -> raise too
        with _patched((pm, "fetch_season_results", lambda client, season, **_: pd.DataFrame())):
            try:
                pm.load_results([2026], client=object(), require_live=[2026])
            except pm.LiveDataError:
                pass
            else:
                raise AssertionError("an empty live payload for a required season must raise")
        # required but no network at all -> raise
        try:
            pm.load_results([2026], client=None, require_live=[2026])
        except pm.LiveDataError:
            pass
        else:
            raise AssertionError("require_live without a client must raise")


def test_starter_top_up_failure_raises_for_required_seasons():
    import requests

    class TimeoutClient:
        def _get_json(self, path, params=None):
            raise requests.Timeout("read timed out")

    with _patched((pm, "LIVE_FETCH_BACKOFF_SECONDS", 0.0)):
        try:
            pm.fetch_starter_lines(TimeoutClient(), {2026: {1, 2}}, require_live=[2026])
        except pm.LiveDataError as exc:
            assert "2026" in str(exc)
        else:
            raise AssertionError("a failed starter top-up for a required season must raise")
        assert pm.fetch_starter_lines(TimeoutClient(), {2022: {1, 2}}, require_live=[2026]).empty


def test_export_raises_and_writes_nothing_when_current_season_fetch_times_out():
    import requests
    import scripts.export_mlb_frontend_data as ex

    current = pd.Timestamp.utcnow().year

    def fake_fetch(client, season, **_):
        if season == current:
            raise requests.Timeout("statsapi.mlb.com read timed out")
        return _raw_results([season]) if season in (2024, 2025) else pd.DataFrame()

    def fake_local(season, normalized_dir=None):
        if season == current:   # stale local table: ends in July, no postseason
            return _raw_results([2025]).assign(season=current)
        raise FileNotFoundError(str(season))

    class NoNetworkClient:
        def _get_json(self, *a, **k):
            raise AssertionError("unexpected network call")

    with tempfile.TemporaryDirectory() as d:
        out, log = Path(d, "out"), Path(d, "picks")
        with _patched((pm, "fetch_season_results", fake_fetch), (pm, "load_local_season", fake_local),
                      (pm, "load_local_starter_lines", lambda *a, **k: pd.DataFrame(columns=pm.LINE_COLUMNS)),
                      (pm, "LIVE_FETCH_BACKOFF_SECONDS", 0.0),
                      (ex, "FRONTEND_DATA_DIR", out), (ex, "MARKET_LOG_ROOT", log),
                      (ex, "MLBStatsClient", NoNetworkClient), (ex, "_ENGINE_CACHE", {})):
            try:
                ex.export_mlb_frontend_data()
            except pm.LiveDataError as exc:
                assert str(current) in str(exc)
            else:
                raise AssertionError("export must fail when the current season cannot be fetched live")
        assert not any(out.rglob("*")), sorted(out.rglob("*"))   # no board, no summary, no temp file
        assert not log.exists() or not any(log.rglob("*"))


def test_apply_engine_predictions_fails_loudly_on_missing_game():
    import scripts.export_mlb_frontend_data as ex

    engine = _market_engine([{"game_pk": 1, "is_final": False, "home_score": None, "away_score": None,
                              "dt": "2026-10-03 20:08", "status_detailed": "Scheduled", "home_win_probability": 0.56,
                              "away_win_probability": 0.44, "p_home": 0.56, "mu_home": 4.4, "mu_away": 4.0,
                              "nb_r": 3.7, "elo_p": 0.55, "train_cutoff": "2026-10-01",
                              "model_version": "mlb-pregame-v1@2026-10-01", "home_team_abbreviation": "LAD",
                              "away_team_abbreviation": "ATL"}])
    upcoming = pd.DataFrame({"game_pk": [1, 2], "home_team_name": ["Dodgers", "Brewers"],
                             "away_team_name": ["Braves", "Padres"]})
    try:
        ex._apply_engine_predictions(upcoming, engine)
    except ex.MissingPredictionError as exc:
        assert "[2]" in str(exc)
    else:
        raise AssertionError("an upcoming game without a prediction must not be dropped silently")
    out = ex._apply_engine_predictions(upcoming.iloc[:1], engine)
    assert len(out) == 1 and out["home_win_probability"].iloc[0] == 0.56


def test_upcoming_schedule_skips_only_undecided_postseason_slots():
    import scripts.export_mlb_frontend_data as ex

    today = pd.Timestamp.utcnow().normalize()

    def game(pk, away, home, state="Scheduled", abstract="Preview"):
        return {"gamePk": pk, "gameType": "L", "season": str(today.year),
                "gameDate": (today + pd.Timedelta(days=1, hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "officialDate": (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "status": {"abstractGameState": abstract, "detailedState": state},
                "teams": {"away": {"team": {"id": away[0], "name": away[1]}},
                          "home": {"team": {"id": home[0], "name": home[1]}}}}

    payload = {"dates": [{"games": [
        game(1, (144, "Atlanta Braves"), (119, "Los Angeles Dodgers")),
        game(2, (5525, "NL Lower Seed"), (5517, "NL Higher Seed")),
        game(3, (147, "New York Yankees"), (139, "Tampa Bay Rays"), state="Postponed", abstract="Final"),
    ]}]}

    class FakeClient:
        def get_schedule(self, **kwargs):
            return payload

    upcoming = ex._load_upcoming_schedule(FakeClient())
    assert upcoming["game_pk"].tolist() == [1]


def test_market_summary_file_is_calibration_only():
    import scripts.export_mlb_frontend_data as ex

    engine = _market_engine(
        [{"game_pk": pk, "is_final": True, "home_score": float(3 + pk % 4), "away_score": float(2 + pk % 3),
          "dt": "2026-09-20 23:05", "status_detailed": "Final"} for pk in range(200, 230)],
        listings=[_listing(pk, "Final", "2026-09-20T23:05:00Z", 3 + pk % 4, 2 + pk % 3) for pk in range(200, 230)],
        live_seasons={2026})
    old_boards = [{"id": f"mlb-{pk}", "gameId": str(pk), "scheduledDate": 20260920, "gameStart": "2026-09-20T23:05:00Z",
                   "homeTeam": "H", "awayTeam": "A", "modelVersion": "v",
                   "markets": mlb_market_picks(board_id=f"mlb-{pk}", home_label="H", away_label="A", p_home=0.55,
                                               mu_home=4.6, mu_away=4.1, nb_r=3.7,
                                               published_at="2026-09-20T07:10:00+00:00", model_version="v")}
                  for pk in range(200, 230)]
    with tempfile.TemporaryDirectory() as d:
        out, log = Path(d, "out"), Path(d, "picks")
        out.mkdir()
        with _patched((ex, "FRONTEND_DATA_DIR", out), (ex, "MARKET_LOG_ROOT", log)):
            ml.write_snapshot("mlb", ml.picks_from_boards("mlb", old_boards, season_of=ml.season_calendar,
                                                          model_version="v",
                                                          published_at="2026-09-20T07:10:00+00:00"), root=log)
            info = ex._write_market_log(engine, [])
        summary = json.loads(Path(out, "mlb_market_summary.json").read_text())
        history = json.loads(Path(out, "mlb_market_history.json").read_text())
    assert info["graded"] == 150 and len(history) == 150
    assert {s["type"] for s in summary} == {"run_line", "total", "team_total_home", "team_total_away"}
    for s in summary:
        assert s["recordKind"] == "calibration", s
        assert s["wins"] is None and s["losses"] is None and s["winRateExPush"] is None and s["roi"] is None
        assert s["hits"] is not None and s["expectedHits"] > 0 and s["brier"] is not None
    assert not list(out.glob("*.tmp"))


def test_record_summary_baselines_and_label():
    import scripts.export_mlb_frontend_data as ex

    rows = []
    for k in range(40):
        rows.append({"game_pk": k, "is_final": True, "home_win": float(k % 3 != 0), "home_win_probability": 0.55 if k % 2 else 0.45,
                     "elo_p": 0.52, "season": 2025, "game_type": "R" if k < 36 else "D",
                     "official_date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=k), "dt": pd.Timestamp("2025-06-01") + pd.Timedelta(days=k)})
    engine = _market_engine(rows)
    engine.params["market_baselines"] = [{"season": 2025, "favoriteWinRate": 0.5637}]
    summary = ex._record_summary(engine)
    assert summary["basis"] == "simulated" and "Simulated" in summary["label"]
    season = summary["seasons"][0]
    assert season["games"] == 40 and season["regularSeasonGames"] == 36 and season["postseasonGames"] == 4
    assert season["regularSeason"]["games"] == 36 and season["alwaysHomeAccuracy"] is not None
    assert summary["marketBaselines"][0]["favoriteWinRate"] == 0.5637


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
        assert len(chosen) == 5 and all(p["publishedAt"].startswith("2026-10-02") for p in chosen.values())
        graded = ml.grade_picks(chosen, {"9": (5.0, 2.0)})
    by_id = {g["marketId"]: g for g in graded}
    assert by_id["mlb-9:run_line:full_game:home"]["result"] == "win"    # CLE -1.5, won by 3
    assert by_id["mlb-9:run_line:full_game:away"]["result"] == "loss"   # CWS -1.5
    assert by_id["mlb-9:total:full_game"]["result"] in ("win", "loss")
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
