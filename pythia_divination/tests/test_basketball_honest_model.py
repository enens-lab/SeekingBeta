"""Checks for the honest NBA/WNBA model, its walk-forward history and the board markets.

Run (cwd = pythia_divination):  python tests/test_basketball_honest_model.py
"""
from __future__ import annotations

import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from sports import market_log as ml  # noqa: E402
from sports import markets as mk  # noqa: E402
from sports.basketball import feature_engineering as fe  # noqa: E402
from sports.basketball import honest_model as hm  # noqa: E402
from sports.basketball.backfill_history import canonical_game_id, game_type_from_id  # noqa: E402
from sports.basketball.elo import EloParams, mov_multiplier, run_elo  # noqa: E402
from sports.basketball.results import add_rest_features, build_results  # noqa: E402

PARAMS = hm.load_params()


def _sched(rows: list[dict]) -> pd.DataFrame:
    base = {"league": "nba", "status_code": 3, "is_neutral": False, "arena_name": "Arena",
            "home_team_city": "H", "away_team_city": "A", "series_text": None}
    out = []
    for r in rows:
        row = {**base, **r}
        row.setdefault("home_team_tricode", f"T{row['home_team_id']}")
        row.setdefault("away_team_tricode", f"T{row['away_team_id']}")
        row.setdefault("home_team_name", f"Team{row['home_team_id']}")
        row.setdefault("away_team_name", f"Team{row['away_team_id']}")
        out.append(row)
    return pd.DataFrame(out)


def test_game_ids_and_types():
    assert canonical_game_id("22100001") == "0022100001"
    assert canonical_game_id(22100001.0) == "0022100001"
    assert canonical_game_id(None) is None
    assert game_type_from_id("0022500001") == "regular"
    assert game_type_from_id("0042500101") == "playoff"
    assert game_type_from_id("0052300101") == "play_in"
    assert game_type_from_id("0062300001") == "cup_final"     # NBA Cup final
    assert game_type_from_id("1052600001") == "cup_final"     # WNBA Commissioner's Cup final, not a play-in
    assert game_type_from_id("1042600305") == "playoff"
    assert game_type_from_id("0012500001") == "preseason"
    assert game_type_from_id("1032600001") == "all_star"


def test_results_merge_prefers_final_rows_and_drops_stale_unplayed():
    stale = _sched([
        {"game_id": "22500001", "game_date_time_utc": "2025-10-21T23:30:00Z", "home_team_id": 1, "away_team_id": 2,
         "status_code": 1, "home_score": 0, "away_score": 0},
        {"game_id": "0022500002", "game_date_time_utc": "2025-10-22T23:30:00Z", "home_team_id": 3, "away_team_id": 4,
         "status_code": 1, "home_score": 0, "away_score": 0},   # never played, in the past -> dropped
        {"game_id": "0012500003", "game_date_time_utc": "2025-10-05T23:30:00Z", "home_team_id": 1, "away_team_id": 3,
         "home_score": 100, "away_score": 90},                   # preseason -> not competitive
    ])
    live = _sched([{"game_id": "0022500001", "game_date_time_utc": "2025-10-21T23:30:00Z", "home_team_id": 1,
                    "away_team_id": 2, "home_score": 110, "away_score": 101}])
    res = build_results("nba", [stale, live], today=pd.Timestamp("2025-11-01"))
    assert list(res["game_id"]) == ["0022500001"], res["game_id"].tolist()
    row = res.iloc[0]
    assert bool(row["final"]) and row["home_score"] == 110 and row["season"] == "2025-26" and row["kind"] == "regular"


def test_rest_uses_local_tip_dates_and_caps():
    # Team 1 plays 18:30 ET on Jan 9 (23:30 UTC) and 21:30 ET on Jan 10 (02:30 UTC Jan 11):
    # a back-to-back on local dates although the UTC dates are two days apart.
    games = _sched([
        {"game_id": "0022500100", "game_date_time_utc": "2026-01-09T23:30:00Z", "home_team_id": 1, "away_team_id": 2,
         "home_score": 100, "away_score": 90},
        {"game_id": "0022500101", "game_date_time_utc": "2026-01-11T02:30:00Z", "home_team_id": 3, "away_team_id": 1,
         "home_score": 100, "away_score": 90},
        {"game_id": "0022500102", "game_date_time_utc": "2026-01-21T00:00:00Z", "home_team_id": 1, "away_team_id": 3,
         "home_score": 100, "away_score": 90},
        # next season opener: rest is not measured across the off-season
        {"game_id": "0022600001", "game_date_time_utc": "2026-10-21T00:00:00Z", "home_team_id": 1, "away_team_id": 2,
         "status_code": 1, "home_score": 0, "away_score": 0},
    ])
    res = add_rest_features(build_results("nba", [games], today=pd.Timestamp("2026-10-01")), cap=4)
    r = res.set_index("game_id")
    assert r.loc["0022500100", "rest_home_c"] == 4 and r.loc["0022500100", "b2b_home"] == 0   # season opener
    assert r.loc["0022500101", "rest_away_days"] == 1 and r.loc["0022500101", "b2b_away"] == 1
    assert r.loc["0022500102", "rest_home_days"] == 10 and r.loc["0022500102", "rest_home_c"] == 4
    assert r.loc["0022600001", "rest_home_c"] == 4 and r.loc["0022600001", "b2b_home"] == 0
    assert r.loc["0022500101", "rest_diff"] == r.loc["0022500101", "rest_home_c"] - r.loc["0022500101", "rest_away_c"]


def test_elo_hand_computed_carry_over_and_pregame_only():
    games = _sched([
        {"game_id": "0022500001", "game_date_time_utc": "2025-10-21T23:30:00Z", "home_team_id": 1, "away_team_id": 2,
         "home_score": 110, "away_score": 100},
        {"game_id": "0022500002", "game_date_time_utc": "2025-10-23T23:30:00Z", "home_team_id": 2, "away_team_id": 1,
         "home_score": 95, "away_score": 99, "is_neutral": True},
        {"game_id": "0022600001", "game_date_time_utc": "2026-10-21T23:30:00Z", "home_team_id": 1, "away_team_id": 2,
         "status_code": 1, "home_score": 0, "away_score": 0},
    ])
    res = build_results("nba", [games], today=pd.Timestamp("2026-10-01"))
    params = EloParams(k=16, hca=60, carry=0.75)
    frame, final = run_elo(res, params)
    f = frame.set_index("game_id")
    p1 = 1 / (1 + 10 ** (-60 / 400))
    assert abs(f.loc["0022500001", "p_elo"] - p1) < 1e-12 and f.loc["0022500001", "elo_home"] == 1500
    delta = 16 * mov_multiplier(10, 60) * (1 - p1)
    assert abs(f.loc["0022500002", "elo_away"] - (1500 + delta)) < 1e-9          # team 1 after its win
    assert abs(f.loc["0022500002", "elo_diff"] - (1500 - delta - (1500 + delta))) < 1e-9   # neutral: no HCA
    # new season: ratings regress 25% toward 1500 before the opener; scheduled games never update
    _, end_of_season = run_elo(res.loc[res["season"] == "2025-26"], params)
    e1 = end_of_season[1]
    assert e1 > 1500 + delta  # won both games
    assert abs(f.loc["0022600001", "elo_home"] - (1500 + 0.75 * (e1 - 1500))) < 1e-9
    assert final[1] == f.loc["0022600001", "elo_home"]
    # pregame only: changing a game's own result never changes its pregame rating
    games2 = games.copy()
    games2.loc[0, ["home_score", "away_score"]] = [80, 120]
    frame2, _ = run_elo(build_results("nba", [games2], today=pd.Timestamp("2026-10-01")), params)
    assert frame2.set_index("game_id").loc["0022500001", "p_elo"] == f.loc["0022500001", "p_elo"]
    assert frame2.set_index("game_id").loc["0022500002", "elo_away"] != f.loc["0022500002", "elo_away"]


def _team_games(n_days: int = 6) -> pd.DataFrame:
    """Two teams, alternating home, across two seasons; team 1 wins every game."""
    rows = []
    for i in range(n_days):
        season = "2024-25" if i < 4 else "2025-26"
        home, away = (1, 2) if i % 2 == 0 else (2, 1)
        home_pts, away_pts = (110, 100) if home == 1 else (100, 110)
        rows.append({
            "league": "nba", "game_id": f"g{i}", "official_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=3 * i)
            + (pd.Timedelta(days=300) if season == "2025-26" else pd.Timedelta(0)),
            "season_display": season, "home_team_id": home, "away_team_id": away,
            "home_team_key": f"nba:{home}", "away_team_key": f"nba:{away}",
            "home_team_name": f"T{home}", "away_team_name": f"T{away}",
            "home_is_winner": 1.0 if home == 1 else 0.0, "away_is_winner": 0.0 if home == 1 else 1.0,
            "home_points": home_pts, "away_points": away_pts,
        })
    return pd.DataFrame(rows)


def test_form_features_reset_each_season_and_include_last_game():
    games = _team_games()
    logs = fe.build_team_game_logs(games)
    feats = fe.attach_pregame_team_features(games, logs).set_index("game_id")
    # game 3 (season 1): team 1 has won games 0,1,2 -> record through the PREVIOUS game is 3-0
    assert feats.loc["g3", "away_team_games_played_prior"] == 3 and feats.loc["g3", "away_team_win_pct_prior"] == 1.0
    # last-3 points include the immediately preceding game (the old double lag skipped it)
    assert feats.loc["g1", "away_team_points_scored_avg_last_3"] == 110.0
    # first game of season 2: no same-season history -> NaN, not last season's 4-0
    assert pd.isna(feats.loc["g4", "home_team_win_pct_prior"]) and pd.isna(feats.loc["g4", "home_team_games_played_prior"])
    assert feats.loc["g5", "away_team_games_played_prior"] == 1
    assert not any("days_rest" in c for c in feats.columns if c.startswith(("home_team_", "away_team_")))


def test_feature_selection_drops_identifiers_and_leaky_standings():
    cols = ["home_team_points_scored_avg_last_5", "away_team_id_detail", "home_team_id", "matchup_diff_wins",
            "matchup_diff_losses", "home_team_wins", "matchup_diff_id_detail", "home_team_days_rest",
            "home_rest_advantage", "matchup_diff_net_rating_est_avg_last_10", "home_availability_core_availability_rating",
            "home_team_key", "month"]
    frame = pd.DataFrame({c: [1.0, 2.0] for c in cols})
    frame["home_team_key"] = ["a", "b"]
    selected = hm.select_feature_columns(frame, PARAMS)
    assert selected == ["home_team_points_scored_avg_last_5", "matchup_diff_net_rating_est_avg_last_10",
                        "home_availability_core_availability_rating"], selected


def _synthetic_design(n_per_season: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s_i, season in enumerate(("2023-24", "2024-25", "2025-26")):
        start = pd.Timestamp(f"{2023 + s_i}-10-20")
        for k in range(n_per_season):
            day = start + pd.Timedelta(days=int(k * 160 / n_per_season))
            elo_diff = rng.normal(60, 140)
            x = rng.normal()
            logit = elo_diff * math.log(10) / 400 + 0.3 * x
            y = float(rng.random() < 1 / (1 + math.exp(-logit)))
            margin = elo_diff / 28 + rng.normal(0, 12)
            rows.append({"game_id": f"{season}-{k}", "season": season, "local_date": day, "final": True,
                         "home_win": y, "elo_diff": elo_diff, "x1": x, "noise": rng.normal(),
                         "rest_home_c": 2.0, "rest_away_c": 2.0, "b2b_home": 0.0, "b2b_away": 0.0,
                         "rest_diff": 0.0, "b2b_diff": 0.0, "is_postseason": 0.0, "is_neutral": 0.0,
                         "margin": margin, "p_elo": 1 / (1 + 10 ** (-elo_diff / 400))})
    return pd.DataFrame(rows).sort_values(["local_date", "game_id"]).reset_index(drop=True)


def test_walk_forward_never_grades_a_game_on_or_before_its_cutoff():
    design = _synthetic_design()
    cols = ["x1", "noise", *PARAMS["win_model"]["extra_features"]]
    preds = hm.walk_forward(design, cols, PARAMS, "nba", win_seasons=["2024-25", "2025-26"], margin_seasons=["2024-25", "2025-26"])
    joined = preds.merge(design[["game_id", "local_date"]], on="game_id")
    assert len(joined) == int((design["season"] != "2023-24").sum())
    assert (joined["local_date"] > joined["train_cutoff"]).all()
    assert (joined["local_date"] >= joined["block_start"]).all()
    assert (joined["local_date"] < joined["block_start"] + pd.Timedelta(days=14)).all()
    assert joined["p_home"].between(0, 1).all() and joined["m_elorest"].notna().all()
    # training size grows block by block (expanding window, earlier dates only)
    sizes = joined.groupby("block_start")["n_train"].first()
    assert sizes.is_monotonic_increasing
    for start, n in sizes.items():
        assert n == int((design["local_date"] < start).sum())


def test_offset_l1_keeps_the_elo_prior_and_finds_real_signal():
    design = _synthetic_design(n_per_season=1500)
    cols = ["elo_diff", "x1", "noise"]
    X = design[cols]
    y = design["home_win"].to_numpy()
    strong = hm.OffsetL1Logistic(C=1e-5).fit(X, y)
    assert set(strong.nonzero_features()) == set()           # everything shrunk: pure Elo + intercept
    p = strong.predict_proba(X)[:, 1]
    assert np.corrcoef(p, design["p_elo"])[0, 1] > 0.999
    weak = hm.OffsetL1Logistic(C=0.5).fit(X, y)
    coefs = weak.nonzero_features()
    assert coefs.get("x1", 0) > 0.15 and abs(coefs.get("noise", 0.0)) < abs(coefs["x1"])
    # beats the plain Elo probability on the data it was told about
    assert hm.log_loss(y, weak.predict_proba(X)[:, 1]) < hm.log_loss(y, design["p_elo"].to_numpy())


def test_shrunk_platt_identity_shrinkage_and_recovery():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 20000)
    lg = np.log(p / (1 - p))
    y = (rng.random(len(p)) < 1 / (1 + np.exp(-(0.3 + 0.5 * lg)))).astype(float)   # overconfident source
    few = hm.ShrunkPlatt(prior_strength=20, min_n=200).fit(p[:150], y[:150])
    assert (few.a, few.b) == (0.0, 1.0) and np.allclose(few.transform(p[:10]), p[:10])   # below min_n: identity
    many = hm.ShrunkPlatt(prior_strength=20, min_n=200, window=20000).fit(p, y)
    assert abs(many.b - 0.5) < 0.06 and abs(many.a - 0.3) < 0.06, many.as_dict()
    stiff = hm.ShrunkPlatt(prior_strength=1e7, min_n=200, window=20000).fit(p, y)
    assert abs(stiff.b - 1.0) < 0.01 and abs(stiff.a) < 0.01                          # strong prior: ~identity
    q = many.transform(np.array([0.2, 0.5, 0.8]))
    assert q[0] < q[1] < q[2] and q[0] > 0.2 and q[2] < 0.8                            # shrinks extremes inward


def test_walk_forward_calibrator_sees_only_earlier_predictions():
    design = _synthetic_design(n_per_season=500)
    cols = ["x1", "noise", *PARAMS["win_model"]["extra_features"]]
    params = {**PARAMS, "calibration": {"prior_strength": 20.0, "min_oos": 100, "window": 100000}}
    preds = hm.walk_forward(design, cols, params, "nba", win_seasons=["2024-25", "2025-26"], margin_seasons=[])
    joined = preds.merge(design[["game_id", "local_date"]], on="game_id")
    for start, g in joined.groupby("block_start"):
        earlier = int((joined["local_date"] < start).sum())
        assert int(g["calib_n"].iloc[0]) == earlier, (start, earlier)
        a, b = float(g["calib_a"].iloc[0]), float(g["calib_b"].iloc[0])
        if earlier < 100:
            assert (a, b) == (0.0, 1.0)
        z = a + b * np.log(g["p_raw"] / (1 - g["p_raw"]))
        assert np.allclose(g["p_home"], 1 / (1 + np.exp(-z)), atol=1e-9)
    assert (joined["local_date"] > joined["train_cutoff"]).all()


def test_win_metrics_counts_a_coin_flip_as_no_pick():
    m = hm.win_metrics(np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.5, 0.5, 0.7, 0.4]))
    assert m["n"] == 4 and m["n_picks"] == 2 and m["accuracy"] == 1.0          # the two 0.5s are not home picks
    assert abs(m["brier"] - np.mean([0.25, 0.25, 0.09, 0.16])) < 1e-12


def test_calibration_metrics_and_paired_brier_diff():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.1, 0.9, 5000)
    y = (rng.random(len(p)) < p).astype(float)
    c = hm.calibration_metrics(y, p)
    assert c["slope_ci_includes_1"] and c["intercept_ci_includes_0"], c
    b = c["bucket_80"]
    assert b["n"] > 0 and abs(b["mean_p"] - b["hit"]) < 0.04
    over = hm.calibration_metrics(y, np.clip(0.5 + 1.6 * (p - 0.5), 0.01, 0.99))
    assert over["slope_ci"][1] < 1.0                                           # overconfident: slope < 1
    d = hm.paired_brier_diff(y, p, np.full(len(p), 0.5))
    assert d["diff"] < 0 and d["ci"][1] < 0                                    # true p beats a coin flip
    same = hm.paired_brier_diff(y, p, p)
    assert same["diff"] == 0.0


def test_published_history_drops_warmup_and_coin_flips():
    from scripts.export_basketball_frontend_data import _published_history
    hist = pd.DataFrame({"game_id": list("abcd"), "p_home": [0.6, 0.5, 0.7, 0.3], "home_win": [1.0, 0.0, 1.0, 1.0],
                         "published": [True, True, False, True]})
    run = hm.LeagueRun(league="nba", design=pd.DataFrame(), feature_cols=[], bundle=None, history=hist,
                       upcoming=pd.DataFrame(), trained_through=None, sigmas={}, timings={})
    assert list(_published_history(run)["game_id"]) == ["a", "d"]


def test_rolling_sigma_uses_only_earlier_dates():
    dates = pd.Series(pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 3 + ["2025-01-03"]))
    resid = pd.Series([1.0, -1.0, 1.0, 10.0, -10.0, 10.0, 0.0])
    sig = hm.rolling_oos_sigma(resid, dates, window=400, min_n=3, default=99.0)
    assert list(sig[:3]) == [99.0] * 3                      # nothing earlier yet
    assert abs(sig[3] - np.std([1, -1, 1], ddof=1)) < 1e-12 and sig[3] == sig[5]   # same-day games share it
    assert abs(sig[6] - np.std([1, -1, 1, 10, -10, 10], ddof=1)) < 1e-12


class _Row:
    home_tricode = "DET"
    away_tricode = "BOS"


def _board_markets(margin: float = 5.3, total: float = 226.4, p_home: float = 0.66) -> list[dict]:
    from scripts.export_basketball_frontend_data import build_board_markets
    pred = pd.Series({"p_home": p_home, "margin_pred": margin, "total_pred": total})
    return build_board_markets("basketball-nba-0022600001", _Row(), pred,
                               {"margin": 14.2, "total": 20.1, "team_total": 12.3},
                               published_at="2026-10-20T07:10:00+00:00", version="test")


def test_board_markets_shape_and_compliance():
    picks = _board_markets()
    ids = [p["marketId"] for p in picks]
    assert len(ids) == len(set(ids)) == 9, ids
    assert all(p["basis"] == "model" and p["modelVersion"] == "test" for p in picks)
    assert all("edge" not in p and "market" not in p for p in picks)      # no lines, no edge (compliance)
    by = {(p["type"], p["side"]): p for p in picks}
    ml_pick = by[("moneyline", "home")]
    assert ml_pick["marketId"].endswith(":moneyline:full_game") and abs(ml_pick["modelProbability"] - 0.66) < 1e-9
    home, away = by[("spread", "home")], by[("spread", "away")]
    assert home["line"] == -5.5 and away["line"] == 5.5 and home["label"] == "DET -5.5" and away["label"] == "BOS +5.5"
    assert home["modelLine"] == -5.3 and away["modelLine"] == 5.3
    assert abs(home["modelProbability"] + away["modelProbability"] - 1) < 1e-3   # x.5 line: no push mass
    assert home["modelProbability"] < 0.5 < away["modelProbability"]           # -5.5 is past the 5.3 mean
    over, under = by[("total", "over")], by[("total", "under")]
    assert over["line"] == under["line"] == 226.5 and abs(over["modelProbability"] + under["modelProbability"] - 1) < 1e-3
    tth, tta = by[("team_total_home", "over")], by[("team_total_away", "over")]
    assert abs(tth["modelLine"] - (226.4 + 5.3) / 2) < 0.06 and abs(tta["modelLine"] - (226.4 - 5.3) / 2) < 0.06
    # no confidence tiers without calibrated per-market cut points (plan review C7)
    assert all("confidenceTier" not in p for p in picks)
    # an underdog home team: the away side is the moneyline pick, lines flip sign
    dog = {(p["type"], p["side"]): p for p in _board_markets(margin=-2.7, p_home=0.4)}
    assert ("moneyline", "away") in dog and dog[("spread", "home")]["line"] == 2.5 and dog[("spread", "away")]["line"] == -2.5
    # the API schema accepts every entry
    SportsMarketPick = _schema_class("SportsMarketPick")
    if SportsMarketPick is None:
        print("  (schema check skipped: pythia_prophecy/api/models.py not found)")
        return
    for p in picks:
        dumped = SportsMarketPick(**p).model_dump()
        assert dumped["type"] == p["type"] and dumped["modelProbability"] == p["modelProbability"]


def _schema_class(name: str):
    """The pydantic class from pythia_prophecy/api/models.py. The full module needs
    email-validator (not in the divination venv), so the market classes are exec'd
    from their source on their own."""
    import ast
    path = Path(__file__).resolve().parents[2] / "pythia_prophecy" / "api" / "models.py"
    if not path.exists():
        return None
    tree = ast.parse(path.read_text())
    wanted = {"SportsMarketPrice", "SportsMarketPick"}
    source = "\n\n".join(ast.get_source_segment(path.read_text(), node) for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name in wanted)
    namespace: dict = {}
    exec("from typing import Optional, List, Dict\nfrom pydantic import BaseModel, Field\n" + source, namespace)
    return namespace.get(name)


def test_market_log_round_trip_grades_unpriced_without_units():
    picks = _board_markets()
    board = {"id": "basketball-nba-0022600001", "gameId": "0022600001", "gameStart": "2026-10-20T23:30:00Z",
             "scheduledDate": 20261020, "homeTeam": "Pistons", "awayTeam": "Celtics", "markets": picks}
    with tempfile.TemporaryDirectory() as d:
        records = ml.picks_from_boards("basketball", [board], season_of=ml.season_cross_year(9), model_version="test",
                                       published_at="2026-10-20T07:10:00+00:00")
        ml.write_snapshot("basketball", records, root=Path(d))
        late = ml.picks_from_boards("basketball", [board], season_of=ml.season_cross_year(9), model_version="test",
                                    published_at="2026-10-21T01:00:00+00:00")   # after tip: never graded
        ml.write_snapshot("basketball", late, root=Path(d))
        chosen = ml.pregame_picks(ml.load_snapshots("basketball", root=Path(d)))
        assert len(chosen) == 9 and all(c["publishedAt"].startswith("2026-10-20T07") for c in chosen.values())
        graded = ml.grade_picks(chosen, {"0022600001": (118.0, 104.0)})   # DET by 14, total 222
        res = {(g["type"], g["side"]): g for g in graded}
        assert res[("spread", "home")]["result"] == "win" and res[("spread", "away")]["result"] == "loss"
        assert res[("total", "under")]["result"] == "win" and res[("moneyline", "home")]["result"] == "win"
        assert all(g["unitReturn"] is None for g in graded)                  # model-only: no invented -110 ROI
        summary = {s["type"]: s for s in ml.summarize(graded)}
        assert summary["spread"]["season"] == "2026-27" and summary["spread"]["roi"] is None


def test_tbd_tips_publish_no_game_start():
    from scripts.export_basketball_frontend_data import tbd_tip_game_ids
    live = pd.DataFrame({
        "game_id": ["1042600113", "1042600201", "0022600001", "0022600002"],
        "status_text": ["9:00 pm ET", "TBD", "7:30 pm ET", None],
        "game_date_time_utc": ["2026-10-03T01:00:00Z", "2026-10-04T04:00:00Z", "2026-10-20T23:30:00Z",
                               "2026-12-25T05:00:00Z"],   # midnight Eastern in winter (EST) is a placeholder too
    })
    assert tbd_tip_game_ids(live) == {"1042600201", "0022600002"}


def test_live_model_line_accuracy_is_mae_per_game():
    from scripts.export_basketball_frontend_data import live_model_line_accuracy
    picks = _board_markets(margin=5.3, total=226.4)
    graded = [{**p, "season": "2026-27", "gameId": "g1", "homeScore": 118.0, "awayScore": 104.0} for p in picks]
    acc = live_model_line_accuracy(graded)
    assert acc["spread"]["2026-27"] == {"n": 1, "mae": round(abs(14 - 5.3), 2)}       # both sides = one game
    assert acc["total"]["2026-27"] == {"n": 1, "mae": round(abs(222 - 226.4), 2)}
    assert acc["team_total_home"]["2026-27"]["n"] == 1 and "moneyline" not in acc


def test_real_data_day_t_results_never_change_day_t_features():
    """Leakage perturbation on the local WNBA tables: rewrite every result, box score
    and player line of one day's games; that day's design rows (features, Elo, rest)
    must not move, while the next games of the same teams must."""
    nd = Path(__file__).resolve().parents[1] / "data" / "sports" / "basketball" / "normalized"
    if not (nd / "schedule_wnba_latest.csv").exists():
        print("  (skipped: no local WNBA tables)")
        return
    tables = hm.load_league_tables(nd, "wnba")
    today = pd.Timestamp("2026-10-02")
    base = hm.build_game_frame(tables, PARAMS, today=today)
    day = pd.Timestamp("2026-07-15")
    ids = set(base.loc[base["final"] & (base["local_date"] == day), "game_id"])
    if not ids:
        day = base.loc[base["final"] & (base["season"] == "2026"), "local_date"].median().normalize()
        ids = set(base.loc[base["final"] & (base["local_date"] == day), "game_id"])
    assert ids

    def design_for(t: hm.LeagueTables) -> pd.DataFrame:
        frame = hm.build_game_frame(t, PARAMS, today=today)
        frame = frame.loc[frame["final"]].reset_index(drop=True)
        bundle = hm.build_features(frame, t, include_rotation=True)
        design, cols = hm.assemble_design(frame, bundle, PARAMS)
        return design.set_index("game_id")[cols + ["p_elo", "elo_diff", "home_wins_before", "away_wins_before"]]

    sched = tables.schedules[0].copy()
    hit = sched["game_id"].isin(ids)
    sched.loc[hit, ["home_score", "away_score"]] = sched.loc[hit, ["away_score", "home_score"]].to_numpy() + [[40, 0]]
    details = tables.details.copy()
    num = [c for c in details.columns if c.startswith(("home_", "away_")) and details[c].dtype.kind in "if"
           and not c.endswith("_id")]
    details.loc[details["game_id"].isin(ids), num] = details.loc[details["game_id"].isin(ids), num] * 1.5 + 3
    players = tables.players.copy()
    for c in ("points", "minutes", "assists", "rebounds_total"):
        players.loc[players["game_id"].isin(ids), c] = pd.to_numeric(players.loc[players["game_id"].isin(ids), c],
                                                                     errors="coerce") * 2 + 5
    pert = hm.LeagueTables(league="wnba", schedules=[sched, *tables.schedules[1:]], details=details, players=players)
    a, b = design_for(tables), design_for(pert)
    rows = sorted(ids)
    pd.testing.assert_frame_equal(a.loc[rows], b.loc[rows], check_exact=False, rtol=1e-9, atol=1e-9)
    later = a.index[(base.set_index("game_id").loc[a.index, "local_date"] > day).to_numpy()]
    diff = (a.loc[later].fillna(-999) != b.loc[later].fillna(-999)).any(axis=1)
    assert diff.sum() > 0      # the perturbation is real: later games see it
    print(f"  perturbed {len(ids)} games on {day.date()}: their rows unchanged, {int(diff.sum())} later rows changed")


def test_real_data_walk_forward_if_available():
    """WNBA end to end on the local tables (skipped when the data is not synced)."""
    nd = Path(__file__).resolve().parents[1] / "data" / "sports" / "basketball" / "normalized"
    if not (nd / "schedule_wnba_latest.csv").exists():
        print("  (skipped: no local WNBA tables)")
        return
    tables = hm.load_league_tables(nd, "wnba")
    run = hm.run_league(tables, PARAMS, today=pd.Timestamp("2026-10-02"), upcoming_ids=set(), with_history=True)
    allpreds = run.history.dropna(subset=["p_home"])
    assert (allpreds["local_date"] > allpreds["train_cutoff"]).all()
    from scripts.export_basketball_frontend_data import _published_history
    hist = _published_history(run)
    assert len(hist) > 300
    assert hist["season"].min() >= PARAMS["leagues"]["wnba"]["history_from_season"]
    assert (allpreds["season"] < PARAMS["leagues"]["wnba"]["history_from_season"]).any()   # warm-up predicted, not shown
    assert run.calibration["n"] > 0
    m = hm.win_metrics(hist["home_win"].to_numpy(float), hist["p_home"].to_numpy(float))
    print("  WNBA walk-forward %s: n=%d acc=%.3f brier=%.4f" % (sorted(hist["season"].unique()), m["n"], m["accuracy"], m["brier"]))
    assert m["brier"] < 0.25   # better than a coin flip
    leaky = [c for c in run.feature_cols if c.endswith(("_wins", "_losses", "_id", "_id_detail")) or "days_rest" in c]
    assert not leaky, leaky


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_basketball_honest_model: all checks passed")
