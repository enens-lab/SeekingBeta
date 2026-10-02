"""Checks for the soccer model fix (P1-2), score-matrix markets (P1-7) and the
walk-forward Track Record + market log wiring in the soccer exporter (P1-5).

Run (cwd = pythia_divination):  python tests/test_soccer_model.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("DATABASE_URL", "postgresql://unused:unused@127.0.0.1:5432/none")
DIV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIV))
sys.path.insert(0, str(DIV / "scripts"))

from sports import market_log as ml  # noqa: E402
from sports import markets as mk  # noqa: E402
from sports.soccer import dixon_coles as dcm  # noqa: E402
from sports.soccer import walkforward as wf  # noqa: E402
from sports.soccer.constants import (  # noqa: E402
    canonical_team_name,
    current_season_start,
    load_model_params,
    resolve_team,
    season_label,
)
from sports.soccer.dixon_coles import DixonColesModel, score_matrices  # noqa: E402

import export_soccer_frontend_data as exp  # noqa: E402

PARAMS = load_model_params()


# ── synthetic league ─────────────────────────────────────────────────────────

def _league(n_teams: int = 16, seasons=(2021, 2022, 2023, 2024, 2025), seed: int = 3,
            home_adv: float = 0.25, intercept: float = 0.15, neutral_share: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Double round robin per season from known attack/defense; weekly rounds."""
    rng = np.random.default_rng(seed)
    teams = [f"team {i:02d}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams); att -= att.mean()
    dfn = rng.normal(0, 0.3, n_teams); dfn -= dfn.mean()
    rows = []
    for s in seasons:
        day = pd.Timestamp(f"{s}-08-15")
        pairs = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        rng.shuffle(pairs)
        for k, (i, j) in enumerate(pairs):
            when = day + pd.Timedelta(days=7 * (k // (n_teams // 2)) + int(rng.integers(0, 3)))
            neutral = rng.random() < neutral_share
            lam = np.exp(intercept + (0 if neutral else home_adv) + att[i] - dfn[j])
            mu = np.exp(intercept + att[j] - dfn[i])
            rows.append({"date": when, "home": teams[i], "away": teams[j], "home_goals": int(rng.poisson(lam)),
                         "away_goals": int(rng.poisson(mu)), "season_start": s, "neutral": neutral,
                         "home_display": teams[i].title(), "away_display": teams[j].title()})
    df = pd.DataFrame(rows).sort_values("date", kind="mergesort").reset_index(drop=True)
    return df, {"attack": dict(zip(teams, att)), "defense": dict(zip(teams, dfn)), "home_adv": home_adv, "intercept": intercept}


# ── model ────────────────────────────────────────────────────────────────────

def test_analytic_gradient_matches_finite_differences():
    from scipy.optimize import check_grad
    df, _ = _league(n_teams=8, seasons=(2024,), neutral_share=0.3)
    captured = {}
    orig = dcm.minimize

    def capture(f, x0, **kw):
        captured["f"] = f
        captured["x0"] = x0
        return orig(f, x0, **kw)

    dcm.minimize = capture
    try:
        DixonColesModel().fit(df.home, df.away, df.home_goals, df.away_goals,
                              weights=np.linspace(0.5, 1, len(df)), l2=2.0, neutral=df.neutral)
    finally:
        dcm.minimize = orig
    f = captured["f"]
    x = captured["x0"] + np.random.default_rng(1).normal(0, 0.1, len(captured["x0"]))
    x[-2] = 0.05  # rho
    err = check_grad(lambda p: f(p)[0], lambda p: f(p)[1], x)
    assert err < 1e-3 * max(1.0, np.linalg.norm(f(x)[1])), err


def test_centering_and_goal_level():
    """The old fit added the mean attack to defense (scaling every rate by
    exp(-2*mean_attack)); now attack/defense are centred, the intercept carries
    the goal level and in-sample expected goals match the data."""
    df, truth = _league()
    w = wf.decay_weights(df.date, PARAMS["timeDecayXiPerDay"])
    m = DixonColesModel().fit(df.home, df.away, df.home_goals, df.away_goals, weights=w, l2=PARAMS["l2"])
    assert m.converged
    assert abs(m.attack.mean()) < 1e-9 and abs(m.defense.mean()) < 1e-9
    lam = np.array([m.rates(h, a)[0] for h, a in zip(df.home, df.away)])
    mu = np.array([m.rates(h, a)[1] for h, a in zip(df.home, df.away)])
    assert abs(np.average(lam, weights=w) - np.average(df.home_goals, weights=w)) < 0.02
    assert abs(np.average(mu, weights=w) - np.average(df.away_goals, weights=w)) < 0.02
    assert abs(m.home_adv - truth["home_adv"]) < 0.08, m.home_adv
    # ratings recovered (shrunk a little by L2)
    est = np.array([m.attack[m._index[t]] for t in truth["attack"]])
    assert np.corrcoef(est, list(truth["attack"].values()))[0, 1] > 0.8


def test_unknown_team_is_league_average():
    df, _ = _league()
    m = DixonColesModel().fit(df.home, df.away, df.home_goals, df.away_goals, l2=PARAMS["l2"])
    lam, mu = m.rates("promoted fc", "another new fc")
    assert abs(lam - np.exp(m.intercept + m.home_adv)) < 1e-12 and abs(mu - np.exp(m.intercept)) < 1e-12
    # a league-average side is between the best and the worst known attack
    known = [m.rates(t, "another new fc")[0] for t in m.teams]
    assert min(known) < lam < max(known)
    assert not m.knows_team("promoted fc")


def test_neutral_flag_fit_matches_predict():
    """Neutral-venue matches must be fitted without home advantage, as they are predicted."""
    df, truth = _league(n_teams=12, neutral_share=0.5, home_adv=0.35)
    aligned = DixonColesModel().fit(df.home, df.away, df.home_goals, df.away_goals, l2=0.5, neutral=df.neutral)
    naive = DixonColesModel().fit(df.home, df.away, df.home_goals, df.away_goals, l2=0.5)
    assert abs(aligned.home_adv - 0.35) < 0.08, aligned.home_adv
    assert naive.home_adv < aligned.home_adv - 0.08, (naive.home_adv, aligned.home_adv)


def test_score_matrix_and_serialisation():
    df, _ = _league(n_teams=10, seasons=(2025,))
    m = DixonColesModel().fit(df.home, df.away, df.home_goals, df.away_goals, l2=2.0)
    mat = m.score_matrix("team 01", "team 02")
    assert mat.shape == (11, 11) and abs(mat.sum() - 1) < 1e-12
    pred = m.predict_match("team 01", "team 02")
    head = mk.score_matrix_markets(mat)
    assert abs(head["home"] - pred["homeWin"]) < 1e-12 and abs(head["draw"] - pred["draw"]) < 1e-12
    batch = score_matrices(np.array([pred["lambdaHome"]]), np.array([pred["lambdaAway"]]), m.rho)[0]
    assert np.allclose(batch, mat)
    clone = DixonColesModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert clone.rates("team 01", "team 02") == m.rates("team 01", "team 02")


# ── season / names ───────────────────────────────────────────────────────────

def test_season_rollover_july():
    assert current_season_start(date(2026, 6, 30)) == 2025
    assert current_season_start(date(2026, 7, 1)) == 2026
    assert current_season_start(date(2027, 1, 15)) == 2026
    assert season_label(2026) == "2026-27"
    assert ml.season_cross_year(7)(20260915) == "2026-27" and ml.season_cross_year(7)(20260520) == "2025-26"


def test_espn_names_resolve_to_football_data():
    known = {canonical_team_name(n) for n in ["Coventry", "Hull", "Nott'm Forest", "FC Koln", "Alaves", "Espanol",
                                              "La Coruna", "Santander", "Paris FC", "Paris SG", "Rennes", "Paderborn",
                                              "Ath Madrid", "M'gladbach", "Man City"]}
    cases = {"Coventry City": "coventry", "Hull City": "hull", "Nottingham Forest": "nott'm forest",
             "FC Cologne": "fc koln", "Alavés": "alaves", "Espanyol": "espanol", "Deportivo": "la coruna",
             "Racing Santander": "santander", "Paris FC": "paris", "Paris Saint-Germain": "paris sg",
             "Stade Rennais": "rennes", "SC Paderborn 07": "paderborn", "Atlético Madrid": "ath madrid",
             "Borussia Mönchengladbach": "m'gladbach", "Manchester City": "man city"}
    for espn, want in cases.items():
        assert resolve_team(espn, known) == want, (espn, resolve_team(espn, known))
    assert resolve_team("Brand New FC", known) not in known
    assert canonical_team_name(float("nan")) == ""


# ── walk-forward ─────────────────────────────────────────────────────────────

def test_refit_cutoff_monday_friday():
    assert wf.refit_cutoff(pd.Timestamp("2026-10-03")) == pd.Timestamp("2026-10-02")   # Sat -> Fri
    assert wf.refit_cutoff(pd.Timestamp("2026-10-05")) == pd.Timestamp("2026-10-05")   # Mon -> Mon
    assert wf.refit_cutoff(pd.Timestamp("2026-10-08")) == pd.Timestamp("2026-10-05")   # Thu -> Mon


def test_walk_forward_never_sees_the_future():
    """Rewriting every result from a date on must not change any prediction for
    a match whose refit cut-off is on or before that date."""
    df, _ = _league(n_teams=10, seasons=(2021, 2022, 2023, 2024, 2025))
    base = wf.walk_forward(df, PARAMS, start_season=2025)
    pivot = pd.Timestamp("2025-10-27")   # a Monday inside the synthetic 2025-26 season
    tampered = df.copy()
    late = tampered.date >= pivot
    tampered.loc[late, "home_goals"] = 9
    tampered.loc[late, "away_goals"] = 0
    again = wf.walk_forward(tampered, PARAMS, start_season=2025)
    early = base.cutoff <= pivot
    assert early.sum() >= 30 and (~early).sum() >= 30, (early.sum(), (~early).sum())
    assert np.allclose(base.loc[early, "lam"], again.loc[early, "lam"])
    assert np.allclose(base.loc[early, "mu"], again.loc[early, "mu"])
    assert not np.allclose(base.loc[~early, "lam"], again.loc[~early, "lam"])
    assert (base.date >= base.cutoff).all()
    scored = wf.score_predictions(base)
    assert scored["model"]["n"] == len(base) and 0 < scored["model"]["rps"] < 0.3


# ── markets ──────────────────────────────────────────────────────────────────

def _matrix(lam=1.75, mu=0.95, rho=-0.06):
    return score_matrices(np.array([lam]), np.array([mu]), rho)[0]


def test_correct_score_grading():
    assert mk.grade_pick("correct_score", "2-1", None, 2, 1)[0] == "win"
    assert mk.grade_pick("correct_score", "2-1", None, 1, 2)[0] == "loss"
    assert mk.grade_pick("correct_score", "0-0", None, 0, 0, decimal_odds=9.0) == ("win", 8.0)
    try:
        mk.grade_pick("correct_score", "home", None, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed correct_score side must raise")


def test_markets_block_is_consistent_with_grading():
    """Every published market's outcome mix equals the matrix-weighted grade_pick
    over all scores, so the board and the graded record can never disagree."""
    m = _matrix()
    picks = exp.build_soccer_markets(board_id="eng.1-1", home_label="Arsenal", away_label="Leeds United",
                                     matrix=m, published_at="2026-10-02T07:10:00+00:00")
    ids = [p["marketId"] for p in picks]
    assert len(ids) == len(set(ids)), ids
    types = [p["type"] for p in picks]
    for t in ("total", "btts", "asian_handicap", "draw_no_bet", "double_chance", "correct_score"):
        assert t in types, t
    assert types.count("total") == 3 and types.count("correct_score") == 3
    assert sorted(p["line"] for p in picks if p["type"] == "total") == [1.5, 2.5, 3.5]
    head = mk.score_matrix_markets(m)
    for p in picks:
        assert "edge" not in p and "market" not in p, p      # show_edge=False, no sportsbook line
        assert p["basis"] == "model" and p["modelVersion"] == exp.MODEL_VERSION
        brute = {k: 0.0 for k in mk.OUTCOMES}
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                brute[mk.grade_pick(p["type"], p["side"], p.get("line"), i, j)[0]] += m[i, j]
        for k in mk.OUTCOMES:
            assert abs(brute[k] - p["outcomeProbabilities"][k]) < 2e-3, (p["label"], k, brute[k], p["outcomeProbabilities"])
    dnb = next(p for p in picks if p["type"] == "draw_no_bet")
    assert dnb["side"] == "home" and abs(dnb["modelProbability"] - head["home"] / (head["home"] + head["away"])) < 1e-3
    fav_lines = sorted(p["line"] for p in picks if p["type"] == "asian_handicap" and p["marketId"].endswith(("fav-0.5", "fav-1.5")))
    assert fav_lines == [-1.5, -0.5]
    fair = exp.fair_handicap_line(m, "home")
    assert abs(mk.score_matrix_handicap(m, "home", fair).win_ex_push - 0.5) < 0.06
    cs = [p["side"] for p in picks if p["type"] == "correct_score"]
    top = mk.correct_scores(m, 3)
    assert cs == [f"{a}-{b}" for a, b, _ in top]
    for p in picks:
        if p["type"] in ("total", "btts"):
            assert p["modelProbability"] >= 0.5   # the more likely side is the one published


def test_board_keeps_client_keys_and_adds_market_fields():
    m = _matrix()
    head = mk.score_matrix_markets(m)
    board = exp.build_soccer_board(
        board_id="eng.1-401879268", name="Arsenal vs Leeds United", tour="Premier League", home_team="Arsenal",
        away_team="Leeds United", home_win_probability=head["home"], draw_probability=head["draw"],
        away_win_probability=head["away"], scheduled_date=20261010, game_start="2026-10-10T11:30:00+00:00",
        lambda_home=1.75, lambda_away=0.95, basis="model", model_version=exp.MODEL_VERSION,
        markets=exp.build_soccer_markets(board_id="eng.1-401879268", home_label="Arsenal", away_label="Leeds United",
                                         matrix=m, published_at="2026-10-02T07:10:00+00:00"))
    for key in ("id", "name", "tour", "scheduledDate", "latestDate", "predictedWinner", "homeTeam", "awayTeam",
                "homeWinProbability", "drawProbability", "awayWinProbability", "gameId", "gameStart", "markets",
                "lambdaHome", "lambdaAway", "basis", "modelVersion"):
        assert key in board, key
    assert [p["rank"] for p in board["predictions"]] == [1, 2, 3]
    assert {"playerName", "winProbability", "side", "rank"} <= set(board["predictions"][0])
    assert board["predictedWinner"] == "Arsenal"
    # API schema round trip (pythia_prophecy/api/models.py), when its deps are installed
    models_py = DIV.parent / "pythia_prophecy" / "api" / "models.py"
    try:
        spec = importlib.util.spec_from_file_location("prophecy_models_for_test", models_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - prophecy deps absent
        print("  (schema round trip skipped: %s)" % exc)
        return
    dumped = mod.SportsUpcomingBoard(**board).model_dump()
    assert len(dumped["markets"]) == len(board["markets"])
    assert {m_["type"] for m_ in dumped["markets"]} >= {"total", "btts", "correct_score"}


# ── exporter: market log + history merge ─────────────────────────────────────

def test_market_log_round_trip_and_grading():
    m = _matrix()
    head = mk.score_matrix_markets(m)

    def board(bid, start, published):
        return exp.build_soccer_board(
            board_id=bid, name="Arsenal vs Leeds United", tour="Premier League", home_team="Arsenal",
            away_team="Leeds United", home_win_probability=head["home"], draw_probability=head["draw"],
            away_win_probability=head["away"], scheduled_date=int(start[:10].replace("-", "")), game_start=start,
            lambda_home=1.75, lambda_away=0.95, basis="model", model_version=exp.MODEL_VERSION,
            markets=exp.build_soccer_markets(board_id=bid, home_label="Arsenal", away_label="Leeds United",
                                             matrix=m, published_at=published))

    history = pd.DataFrame({"date": [pd.Timestamp("2026-09-19")], "home": ["arsenal"], "away": ["leeds"],
                            "home_goals": [2], "away_goals": [0], "season_start": [2026]})
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # bake 1 (Thu): two games; game B result only in football-data
        b1 = [board("eng.1-A", "2026-09-20T14:00:00+00:00", "2026-09-17T07:10:00+00:00"),
              board("eng.1-B", "2026-09-19T14:00:00+00:00", "2026-09-17T07:10:00+00:00")]
        exp.log_and_grade_markets(b1, {}, {}, published_at="2026-09-17T07:10:00+00:00", root=root)
        # bake 2 (after game A kicked off): game A's picks are published too late and must be ignored
        b2 = [board("eng.1-A", "2026-09-20T14:00:00+00:00", "2026-09-20T15:00:00+00:00")]
        graded, summary = exp.log_and_grade_markets(
            b2, {"eng.1-A": (1, 1)}, {"eng.1": history}, published_at="2026-09-20T15:00:00+00:00", root=root)
        n_per_game = len(b1[0]["markets"])
        assert len(graded) == 2 * n_per_game, len(graded)
        assert all(g["publishedAt"].startswith("2026-09-17") for g in graded)
        assert all(g["unitReturn"] is None for g in graded)            # unpriced: no invented ROI
        by_game = {g["gameId"]: (g["homeScore"], g["awayScore"]) for g in graded}
        assert by_game == {"eng.1-A": (1, 1), "eng.1-B": (2, 0)}
        cs_a = [g for g in graded if g["gameId"] == "eng.1-A" and g["type"] == "correct_score"]
        assert sum(g["result"] == "win" for g in cs_a) == int(any(g["side"] == "1-1" for g in cs_a))
        types = {s["type"] for s in summary}
        assert {"total", "btts", "asian_handicap", "draw_no_bet", "double_chance", "correct_score"} <= types
        assert all(s["roi"] is None and s["unitsAtStatedPrice"] is None for s in summary)
        assert all(s["season"] == "2026-27" for s in summary)
        # No declared lean: every soccer type is a calibration row, never a W-L record
        # (totals at three lines and handicap at -0.5/-1.5/fair are pooled per type).
        assert all(s["recordKind"] == "calibration" for s in summary), {s["type"]: s["recordKind"] for s in summary}
        for s in summary:
            assert s["wins"] is None and s["losses"] is None and s["winRateExPush"] is None, s
            assert s["breakEvenRate"] is None, s
            assert s["graded"] > 0 and s["hits"] >= 0 and s["expectedHits"] > 0 and s["brier"] is not None, s
        totals = next(s for s in summary if s["type"] == "total")
        assert totals["graded"] == 2 * 3                     # two games x three lines, one row
        assert abs(totals["expectedHits"] - sum(g["modelProbability"] for g in graded if g["type"] == "total")) < 0.01
        assert len(list((root / "soccer").glob("picks_*.jsonl"))) == 2


def test_history_merge_keeps_published_world_cup_and_drops_legacy_league_boards():
    wc_board = {"tour": "FIFA World Cup", "tournament": "Argentina vs Switzerland", "scheduledDate": 20260712,
                "latestDate": 20260712, "hitStatus": "Top Pick", "prob": 0.583}
    rescored = dict(wc_board, hitStatus="Miss", prob=0.41, recordVersion="worldcup-pretournament:x")
    legacy = {"tour": "Premier League", "tournament": "Man City vs Arsenal", "scheduledDate": 20260517,
              "latestDate": 20260517, "hitStatus": "Top Pick"}
    kept = {"tour": "Serie A", "tournament": "Inter vs Milan", "scheduledDate": 20260301, "latestDate": 20260301,
            "hitStatus": "Miss", "recordVersion": exp.RECORD_VERSION}
    new = {"tour": "Premier League", "tournament": "Arsenal vs Leeds", "scheduledDate": 20260920,
           "latestDate": 20260920, "hitStatus": "Miss", "recordVersion": exp.RECORD_VERSION}
    out = exp.merge_history([new, rescored], archive=[wc_board], previous=[legacy, kept])
    keys = [(b["tour"], b["tournament"]) for b in out]
    assert ("Premier League", "Man City vs Arsenal") not in keys
    assert ("Serie A", "Inter vs Milan") in keys and ("Premier League", "Arsenal vs Leeds") in keys
    wc_out = [b for b in out if b["tour"] == "FIFA World Cup"]
    assert len(wc_out) == 1 and wc_out[0]["prob"] == 0.583 and wc_out[0]["hitStatus"] == "Top Pick"
    assert [b["latestDate"] for b in out] == sorted([b["latestDate"] for b in out], reverse=True)


def test_world_cup_archive_is_the_published_record():
    boards = exp.load_world_cup_archive()
    assert len(boards) == 100
    root_archive = DIV.parent / "archive" / "soccer_worldcup2026_historical_backtests_20260712.json"
    if root_archive.exists():
        published = json.loads(root_archive.read_text())
        mine = {(b["tournament"], b["scheduledDate"]): b for b in boards}
        for b in published:   # every archived board is carried verbatim
            assert mine[(b["tournament"], b["scheduledDate"])] == b
    assert all(b.get("hitStatus") in ("Top Pick", "Miss") for b in boards)


def test_history_season_count():
    assert exp._history_seasons(2026) == 6      # 2021-22 .. 2026-27
    assert exp._history_seasons(2027) == 7
    assert exp._record_window_seasons(2026) == [2021, 2022, 2023, 2024, 2025, 2026]
    assert exp._fit_window_seasons(2026) == [2022, 2023, 2024, 2025, 2026]


# ── incomplete football-data history (partial downloads) ─────────────────────

TODAY = date(2026, 10, 2)
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
LEAGUES = {"eng.1": ("E0", 11), "esp.1": ("SP1", 12)}     # league -> (fd code, synthetic seed)


class _patched:
    """Temporarily set attributes (plain-function tests, no pytest fixtures)."""

    def __init__(self, *triples):
        self.triples = triples
        self.saved = []

    def __enter__(self):
        for obj, name, value in self.triples:
            self.saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


def _raw_seasons() -> dict[tuple[str, int], pd.DataFrame]:
    """football-data shaped CSV frames per (fd code, season) for two synthetic leagues."""
    out = {}
    for fd_code, seed in LEAGUES.values():
        df, _ = _league(n_teams=8, seasons=SEASONS, seed=seed)
        df = df[df.date < pd.Timestamp(TODAY)]
        for s, g in df.groupby("season_start"):
            out[(fd_code, int(s))] = pd.DataFrame({
                "Date": g.date.dt.strftime("%d/%m/%Y"), "HomeTeam": g.home_display, "AwayTeam": g.away_display,
                "FTHG": g.home_goals, "FTAG": g.away_goals})
    return out


def _fake_fixtures(league_key, today):
    return [{"id": f"{league_key}-fx{i}", "name": "", "date_int": 20261010, "venue": None, "completed": False,
             "state": "pre", "home_team": {}, "away_team": {}, "home_name": home, "away_name": away,
             "home_score": None, "away_score": None, "start_iso": "2026-10-10T14:00:00+00:00",
             "neutral_site": False, "status_name": "STATUS_SCHEDULED"}
            for i, (home, away) in enumerate((("Team 01", "Team 02"), ("Team 03", "Team 04")))]


def _soccer_env(fail: set, calls: dict):
    """Patch the exporter onto the synthetic leagues; `fail` = {(fd code, season)} whose
    download raises, `calls` collects which leagues were walked forward / refitted."""
    import requests
    from sports.soccer import client as soccer_client

    raw = _raw_seasons()

    def fetch(fd_code, season_start, cache_dir=None, cache=True):
        if (fd_code, int(season_start)) in fail:
            raise requests.ConnectionError(f"simulated outage {fd_code} {season_start}")
        return raw[(fd_code, int(season_start))].copy()

    real_record, real_fit = exp._build_track_record_for_league, exp._fit_current_model

    def record(league_key, history):
        calls.setdefault("record", []).append(league_key)
        return real_record(league_key, history)

    def fit(history, season):
        calls.setdefault("fit", []).append(int(history["season_start"].nunique()))
        return real_fit(history, season)

    return _patched((soccer_client, "fetch_football_data_csv", fetch),
                    (exp, "PREDICTABLE_LEAGUE_KEYS", tuple(LEAGUES)),
                    (exp, "WORLD_CUP_ENABLED", False),
                    (exp, "_fetch_league_fixtures", _fake_fixtures),
                    (exp, "_build_track_record_for_league", record),
                    (exp, "_fit_current_model", fit))


def _league_boards(boards, tour):
    return [b for b in boards if b.get("tour") == tour]


def _published(fail=frozenset()):
    """One clean bake, round-tripped through JSON like the file on disk, with every
    league board stamped so a regenerated board is distinguishable from a carried one."""
    calls: dict = {}
    with _soccer_env(set(fail), calls):
        payload, _, _, health = exp._build_payload(published_at="2026-10-02T07:00:00+00:00", today=TODAY)
    assert not health["frozenRecord"] and sorted(health["pricedLeagues"]) == sorted(LEAGUES)
    history = json.loads(json.dumps(payload["completed"], default=str))
    for b in history:
        if b.get("tour") in ("Premier League", "La Liga"):
            b["publishedRun"] = "previous"
    return history, json.loads(json.dumps(payload["trackRecord"], default=str))


def test_failed_old_season_download_keeps_that_leagues_published_record():
    """E0 2022-23 fails to download: the Premier League is not refitted, its published
    boards stay byte-identical, its upcoming boards are withheld (2022-23 is inside the
    current 5-season window); La Liga still regenerates and prices normally."""
    previous, previous_record = _published()
    prev_pl = _league_boards(previous, "Premier League")
    assert len(prev_pl) > 50 and len(_league_boards(previous, "La Liga")) > 50

    calls: dict = {}
    with _soccer_env({("E0", 2022)}, calls):
        payload, _, histories, health = exp._build_payload(previous_history=previous, previous_track_record=previous_record,
                                                           published_at="2026-10-02T07:10:00+00:00", today=TODAY)
        # the hazard being guarded: a refit on the partial history moves the record
        partial = histories["eng.1"]
        assert 2022 not in set(partial.season_start)
        moved, _ = exp._build_track_record_for_league("eng.1", partial)
    calls["record"].pop()   # the explicit call above

    assert health["frozenRecord"] == {"eng.1": [2022]}
    assert health["unpricedUpcoming"] == {"eng.1": [2022]}
    assert health["pricedLeagues"] == ["esp.1"]
    assert calls["record"] == ["esp.1"] and len(calls["fit"]) == 1          # eng.1 never refitted

    out_pl = _league_boards(payload["completed"], "Premier League")
    assert [json.dumps(b, sort_keys=True) for b in out_pl] == [json.dumps(b, sort_keys=True) for b in prev_pl]
    assert all(b.get("publishedRun") == "previous" for b in out_pl)
    prev_probs = {(b["tournament"], b["scheduledDate"]): b["prob"] for b in prev_pl}
    assert any(prev_probs.get((b["tournament"], b["scheduledDate"])) != b["prob"] for b in moved)

    out_liga = _league_boards(payload["completed"], "La Liga")
    assert len(out_liga) == len(_league_boards(previous, "La Liga"))
    assert all("publishedRun" not in b for b in out_liga)                   # regenerated

    tours = {b["tour"] for b in payload["upcoming"]}
    assert tours == {"La Liga"}, tours
    tr = payload["trackRecord"]
    assert [r for r in tr["leagues"] if r["tour"] == "Premier League"] == \
        [r for r in previous_record["leagues"] if r["tour"] == "Premier League"]
    assert tr["allLeagues"] == previous_record["allLeagues"]
    assert {r["tour"] for r in tr["leagues"]} == {"Premier League", "La Liga"}


def test_gap_outside_the_current_window_freezes_the_record_but_still_prices():
    """2021-22 only feeds the 2025-26 walk-forward, not today's 2022..2026 fit."""
    previous, previous_record = _published()
    calls: dict = {}
    with _soccer_env({("E0", 2021)}, calls):
        payload, _, _, health = exp._build_payload(previous_history=previous, previous_track_record=previous_record,
                                                   published_at="2026-10-02T07:10:00+00:00", today=TODAY)
    assert health["frozenRecord"] == {"eng.1": [2021]} and health["unpricedUpcoming"] == {}
    assert calls["record"] == ["esp.1"] and len(calls["fit"]) == 2
    assert {b["tour"] for b in payload["upcoming"]} == {"Premier League", "La Liga"}
    assert _league_boards(payload["completed"], "Premier League") == _league_boards(previous, "Premier League")


def test_full_outage_keeps_every_record_and_exits_non_zero():
    previous, previous_record = _published()
    outage = {(code, s) for code, _ in LEAGUES.values() for s in SEASONS}
    with tempfile.TemporaryDirectory() as d:
        out, picks = Path(d) / "out", Path(d) / "picks"
        out.mkdir()
        (out / "soccer_historical_backtests.json").write_text(json.dumps(previous, indent=2))
        (out / "soccer_track_record.json").write_text(json.dumps(previous_record, indent=2))
        before = (out / "soccer_historical_backtests.json").read_text()
        with _soccer_env(outage, {}), _patched((exp, "_market_picks_root", lambda: picks)):
            rc = exp.export_soccer_frontend_data(out, today=TODAY)
        assert rc == 1
        after = json.loads((out / "soccer_historical_backtests.json").read_text())
        assert json.loads(before) == after
        assert json.loads((out / "soccer_track_record.json").read_text())["leagues"] == previous_record["leagues"]
        assert json.loads((out / "soccer_upcoming_tournaments.json").read_text()) == []


def test_frozen_league_without_a_previous_file_leaves_the_published_record_untouched():
    """A fresh RunPod worker has no previous export on disk: rewriting the record files
    there would drop the frozen league, so they are not written at all."""
    with tempfile.TemporaryDirectory() as d:
        out, picks = Path(d) / "out", Path(d) / "picks"
        with _soccer_env({("E0", 2023)}, {}), _patched((exp, "_market_picks_root", lambda: picks)):
            rc = exp.export_soccer_frontend_data(out, today=TODAY)
        assert rc == 0                                   # La Liga priced honestly
        assert not (out / "soccer_historical_backtests.json").exists()
        assert not (out / "soccer_track_record.json").exists()
        upcoming = json.loads((out / "soccer_upcoming_tournaments.json").read_text())
        assert upcoming and {b["tour"] for b in upcoming} == {"La Liga"}
        summary = json.loads((out / "soccer_market_summary.json").read_text())
        assert summary == []                             # nothing graded yet on a fresh log


def test_current_season_not_published_is_only_a_gap_after_the_grace_date():
    import requests
    from sports.soccer import client as soccer_client

    frame = pd.DataFrame({"Date": ["16/08/2025"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Leeds"], "FTHG": [1], "FTAG": [0]})

    def fetch(fd_code, season_start, cache_dir=None, cache=True):
        if season_start == 2026:
            resp = requests.Response(); resp.status_code = 404
            raise requests.HTTPError("404 Client Error", response=resp)
        if season_start == 2024:
            raise requests.Timeout("read timed out")
        if season_start == 2023:
            return frame.iloc[0:0]                          # header only
        return frame.assign(Date=f"16/08/{season_start}")

    with _patched((soccer_client, "fetch_football_data_csv", fetch)):
        load = soccer_client.load_history_report("eng.1", seasons=6, current_season_start=2026)
    assert load.loaded == (2021, 2022, 2025)
    assert load.not_published == (2026,) and load.empty == (2023,) and list(load.failed) == [2024]
    window = [2022, 2023, 2024, 2025, 2026]
    assert exp._missing_seasons(load, window, season=2026, today=date(2026, 8, 10)) == [2023, 2024]
    assert exp._missing_seasons(load, window, season=2026, today=date(2026, 9, 15)) == [2023, 2024, 2026]
    assert exp._missing_seasons(None, window, season=2026, today=date(2026, 8, 10)) == window
    timeout = soccer_client.HistoryLoad(frame=pd.DataFrame(), requested=(2026,), loaded=(), failed={2026: "Timeout"})
    assert exp._missing_seasons(timeout, [2026], season=2026, today=date(2026, 8, 10)) == [2026]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("test_soccer_model: all checks passed")
