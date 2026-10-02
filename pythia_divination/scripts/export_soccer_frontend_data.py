"""Soccer (association football) frontend data export.

Builds the soccer board payload the BFF serves (static JSON, refreshed by the
RunPod worker) and the divination ``/api/sports/soccer/boards`` fallback.

Soccer is ONE sport key whose ``tour`` carries the competition (Premier League,
La Liga, Serie A, Bundesliga, Ligue 1, FIFA World Cup), mirroring golf/tennis.

Pipeline per league (sports/soccer/model_params.json holds the settings):
  1. load match history (football-data.co.uk) for the current season -- derived
     from today's date, July rollover -- and the 5-season window behind the record;
  2. TRACK RECORD: a walk-forward over every match since the 2025-26 season start,
     each predicted by a Dixon-Coles fit on matches dated strictly before that
     match's refit cut-off (Mon + Fri). The record starts at a fixed season, so it
     only ever grows (no newest-N window), and it is scored next to the de-vigged
     football-data closing market in soccer_track_record.json;
  3. UPCOMING: fit on everything played so far and price each ESPN fixture from
     the full score matrix: 1X2, expected goals, totals 1.5/2.5/3.5, both teams to
     score, Asian handicap (model fair line + favourite -0.5/-1.5), draw no bet,
     double chance and the three most likely scores. ``basis`` is "model": no
     sportsbook line is shown (football-data odds licence unconfirmed; ESPN/DraftKings
     lines are not licensed) and no edge is ever computed (show_edge=False).
  4. MARKET LOG: every published market is appended to the pick log
     (sports/market_log.py); picks published before kick-off are graded against the
     final score into soccer_market_history.json / soccer_market_summary.json. Those
     results never touch hitStatus / seasonSummary (clients count any non-"Miss" as a
     hit and those numbers are emailed).

The World Cup keeps its own record: the boards as published (sports/soccer/archive/,
100 matches 2026-06-11..07-12) plus any later finished match scored by the
pre-tournament international model. It is reported separately from the leagues.

DRAW / 1X2 CONTRACT: each board carries homeWinProbability / drawProbability /
awayWinProbability plus a ``predictions`` mirror of [Home, Draw, Away] rows so
older clients still render something.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

DIV_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports import market_log  # noqa: E402
from sports import markets as mk  # noqa: E402
from sports.soccer import client  # noqa: E402
from sports.soccer.branding import branding_from_espn_team  # noqa: E402
from sports.soccer.constants import (  # noqa: E402
    LEAGUE_CONFIGS,
    PREDICTABLE_LEAGUE_KEYS,
    SEASON_START_MONTH,
    canonical_national_name,
    current_season_start,
    load_model_params,
    resolve_team,
    season_label,
    tour_for_league,
)
from sports.soccer.dixon_coles import DixonColesModel  # noqa: E402
from sports.soccer import walkforward as wf  # noqa: E402
from sports.soccer import world_cup as wc  # noqa: E402
from sports.soccer import world_cup_detail as wcd  # noqa: E402

logger = logging.getLogger(__name__)

LIVE_SOURCE = "divination_live_soccer_feed"
SPORT_KEY = "soccer"
UPCOMING_DAYS_AHEAD = 21
RESULTS_DAYS_BACK = 14          # ESPN finished fixtures fetched for grading the pick log
MARKET_HISTORY_CAP = 300

MODEL_PARAMS = load_model_params()
MODEL_VERSION = str(MODEL_PARAMS["modelVersion"])
RECORD_VERSION = f"walkforward:{MODEL_VERSION}"
RECORD_START_SEASON = int(MODEL_PARAMS["recordStartSeason"])
TIME_DECAY_XI = float(MODEL_PARAMS["timeDecayXiPerDay"])  # per day; ~1-year half life
MAX_GOALS = int(MODEL_PARAMS.get("maxGoals", 10))
TOTALS_LINES = [float(x) for x in MODEL_PARAMS.get("totalsLines", [1.5, 2.5, 3.5])]
HANDICAP_MAIN_LINES = [float(x) for x in MODEL_PARAMS.get("handicapMainLines", [-0.5, -1.5])]
SEASON_OF = market_log.season_cross_year(SEASON_START_MONTH)
MARKET_PICKS_ROOT_ENV = "SPORTS_MARKET_PICKS_ROOT"

WORLD_CUP_ENABLED = True
# 2000 sims gives the same top-team title odds as 10000 (verified: ±0.5pp) at a
# fraction of the cost — keeps the soccer recompute light on the 2-core box.
WORLD_CUP_SIMS = 2000
WORLD_CUP_MAX_FIXTURES = 32  # cap upcoming WC match boards in the feed
WORLD_CUP_WINDOW = ("20260611", "20260719")  # tournament dates
# The World Cup 2026 record as published: the 90 boards of the repo-root
# archive/soccer_worldcup2026_historical_backtests_20260712.json (unchanged) plus the
# 10 opening matches (2026-06-11..14) that the old 90-board cap had already pushed
# out of that archive but that were published in the 2026-06-18 soccer history file.
# Kept inside pythia_divination so the RunPod image (which copies only this
# directory) carries it.
WORLD_CUP_ARCHIVE = DIV_ROOT / "sports" / "soccer" / "archive" / "worldcup2026_published_backtests.json"

DISCLAIMER = (
    "Probabilities are model estimates for information only. SeekingBeta does not accept, "
    "place or facilitate bets. Past results do not predict future results."
)

_HANDICAP_GRID = [round(x, 2) for x in np.arange(-4.0, 4.0001, 0.25)]
_TOTAL_GRID = [round(x, 2) for x in np.arange(0.5, 7.0001, 0.25)]


# ── board builders ────────────────────────────────────────────────────────────

def build_soccer_board(
    *,
    board_id: str,
    name: str,
    tour: str,
    home_team: str,
    away_team: str,
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
    scheduled_date: int | None = None,
    venue: str | None = None,
    home_team_details: dict[str, Any] | None = None,
    away_team_details: dict[str, Any] | None = None,
    game_id: str | None = None,
    game_start: str | None = None,
    lambda_home: float | None = None,
    lambda_away: float | None = None,
    markets: list[dict[str, Any]] | None = None,
    basis: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Assemble a single draw-aware soccer upcoming board dict.

    Probabilities are 0..1; the ``predictions`` mirror uses 0..100 percentages
    to match the existing ranked-field convention.
    """
    outcomes = [
        (home_team, home_win_probability, "home"),
        ("Draw", draw_probability, "draw"),
        (away_team, away_win_probability, "away"),
    ]
    ranked = sorted(outcomes, key=lambda item: item[1], reverse=True)
    predicted_winner = ranked[0][0]
    predictions = [
        {
            "rank": idx + 1,
            "playerName": label,
            "winProbability": round(prob * 100.0, 2),
            "side": side,
        }
        for idx, (label, prob, side) in enumerate(ranked)
    ]

    board: dict[str, Any] = {
        "id": board_id,
        "gameId": game_id or board_id,
        "gameStart": game_start,
        "name": name,
        "tour": tour,
        "course": "",
        "venue": venue,
        "scheduledDate": scheduled_date,
        "latestDate": scheduled_date,
        "homeTeam": home_team,
        "awayTeam": away_team,
        "predictedWinner": predicted_winner,
        "homeWinProbability": round(float(home_win_probability), 4),
        "drawProbability": round(float(draw_probability), 4),
        "awayWinProbability": round(float(away_win_probability), 4),
        "predictions": predictions,
        "markets": list(markets or []),
    }
    if lambda_home is not None and lambda_away is not None:
        # Expected goals behind every market on the board (Dixon-Coles rates).
        board["lambdaHome"] = round(float(lambda_home), 3)
        board["lambdaAway"] = round(float(lambda_away), 3)
    if basis:
        board["basis"] = basis
    if model_version:
        board["modelVersion"] = model_version
    if home_team_details:
        board["homeTeamDetails"] = home_team_details
    if away_team_details:
        board["awayTeamDetails"] = away_team_details
    return board


def build_soccer_backtest(
    *,
    tour: str,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    prediction: dict[str, float],
    match_date: int | None,
    year: int,
) -> dict[str, Any]:
    """Assemble a completed (historical) soccer board with hit/miss vs actual."""
    outcomes = [
        (home_team, prediction["homeWin"], "home"),
        ("Draw", prediction["draw"], "draw"),
        (away_team, prediction["awayWin"], "away"),
    ]
    ranked = sorted(outcomes, key=lambda item: item[1], reverse=True)
    predicted_label, predicted_prob, _ = ranked[0]

    if home_goals > away_goals:
        actual_label = home_team
    elif away_goals > home_goals:
        actual_label = away_team
    else:
        actual_label = "Draw"

    hit_status = "Top Pick" if predicted_label == actual_label else "Miss"

    return {
        "year": int(year),
        "tournament": f"{home_team} vs {away_team}",
        "tour": tour,
        "hitStatus": hit_status,
        "predictedWinner": predicted_label,
        "actualWinner": f"{actual_label} ({home_goals}-{away_goals})",
        "prob": round(float(predicted_prob), 4),
        "homeWinProbability": round(float(prediction["homeWin"]), 4),
        "drawProbability": round(float(prediction["draw"]), 4),
        "awayWinProbability": round(float(prediction["awayWin"]), 4),
        "homeTeam": home_team,
        "awayTeam": away_team,
        "scheduledDate": match_date,
        "latestDate": match_date,
        "homeScore": int(home_goals),
        "awayScore": int(away_goals),
    }


# ── markets from the score matrix ────────────────────────────────────────────

def _fmt_line(line: float) -> str:
    return "0" if abs(line) < 1e-9 else f"{line:+g}"


def fair_handicap_line(matrix: np.ndarray, side: str = "home") -> float:
    """Quarter-grid handicap (that side's view) whose push-excluded cover
    probability is closest to 50%: the model's own Asian-handicap line."""
    return min(_HANDICAP_GRID, key=lambda line: (
        abs(mk.score_matrix_handicap(matrix, side, line).win_ex_push - 0.5), abs(line)))


def fair_total_line(matrix: np.ndarray) -> float:
    return min(_TOTAL_GRID, key=lambda line: (
        abs(mk.score_matrix_total(matrix, "over", line).win_ex_push - 0.5), line))


def _binary_prices(p: float) -> mk.LinePrices:
    p = float(min(max(p, 0.0), 1.0))
    return mk.LinePrices(win=p, half_win=0.0, push=0.0, half_loss=0.0, loss=1.0 - p)


def build_soccer_markets(*, board_id: str, home_label: str, away_label: str, matrix: np.ndarray,
                         published_at: str, model_version: str = MODEL_VERSION) -> list[dict[str, Any]]:
    """The `markets` block for one fixture, all priced from the same score matrix.

    One entry per market line, on the side the model makes more likely (the line
    markets are fixed: totals 1.5/2.5/3.5, the favourite at -0.5/-1.5), so a logged
    pick's hit rate reads as the model's directional accuracy. marketId carries a
    stable variant suffix: several lines of one market type live on one board."""
    m = np.asarray(matrix, dtype=float)
    m = m / m.sum()
    head = mk.score_matrix_markets(m)
    p_home, p_draw, p_away = head["home"], head["draw"], head["away"]
    team = {"home": home_label, "away": away_label}
    fav = "home" if p_home >= p_away else "away"
    out: list[dict[str, Any]] = []

    def add(variant: str, market_type: str, side: str, label: str, prices: mk.LinePrices,
            line: Optional[float] = None, model_line: Optional[float] = None) -> None:
        entry = mk.market_pick(board_id=board_id, market_type=market_type, side=side, label=label,
                               prices=prices, line=line, model_line=model_line, period="full_game",
                               market=None, published_at=published_at, show_edge=False)
        entry["marketId"] = f"{board_id}:{market_type}:full_game:{variant}"
        entry["basis"] = "model"
        entry["modelVersion"] = model_version
        out.append(entry)

    # Totals (over/under goals) at the standard lines, plus the model's fair total.
    total_fair = fair_total_line(m)
    for line in TOTALS_LINES:
        over = mk.score_matrix_total(m, "over", line)
        side = "over" if over.win_ex_push >= 0.5 else "under"
        prices = over if side == "over" else mk.score_matrix_total(m, "under", line)
        add(f"{line:g}", "total", side, f"{side.title()} {line:g} goals", prices, line=line, model_line=total_fair)

    # Both teams to score.
    btts = head["btts_yes"]
    side = "yes" if btts >= 0.5 else "no"
    add("btts", "btts", side, side.title(), _binary_prices(btts if side == "yes" else 1 - btts))

    # Asian handicap: the favourite at the main lines, and the model's fair line.
    home_fair = fair_handicap_line(m, "home")
    fav_fair = home_fair if fav == "home" else -home_fair
    for line in HANDICAP_MAIN_LINES:
        add(f"fav{line:+g}", "asian_handicap", fav, f"{team[fav]} {_fmt_line(line)}",
            mk.score_matrix_handicap(m, fav, line), line=line, model_line=fav_fair)
    if home_fair < 0:
        fair_side, fair_line = "home", home_fair
    elif home_fair > 0:
        fair_side, fair_line = "away", -home_fair
    else:
        fair_side, fair_line = fav, 0.0
    # Skip the fair-line entry when it duplicates a main line or draw-no-bet (AH 0).
    duplicate = fair_side == fav and any(abs(fair_line - x) < 1e-9 for x in HANDICAP_MAIN_LINES + [0.0])
    if not duplicate:
        add("fair", "asian_handicap", fair_side, f"{team[fair_side]} {_fmt_line(fair_line)}",
            mk.score_matrix_handicap(m, fair_side, fair_line), line=fair_line, model_line=fair_line)

    # Draw no bet (stake returned on a draw) for the more likely winner.
    dnb_side = fav
    add("dnb", "draw_no_bet", dnb_side, f"{team[dnb_side]} (draw no bet)", mk.score_matrix_handicap(m, dnb_side, 0.0))

    # Double chance: the most likely of 1X / X2 / 12.
    options = {
        "home_draw": (p_home + p_draw, f"{home_label} or draw"),
        "away_draw": (p_away + p_draw, f"{away_label} or draw"),
        "home_away": (p_home + p_away, f"{home_label} or {away_label}"),
    }
    dc_side = max(options, key=lambda k: options[k][0])
    add("dc", "double_chance", dc_side, options[dc_side][1], _binary_prices(options[dc_side][0]))

    # The three most likely exact scores (home goals first).
    for rank, (hg, ag, p) in enumerate(mk.correct_scores(m, 3), start=1):
        add(f"cs{rank}", "correct_score", f"{hg}-{ag}", f"Correct score {hg}-{ag}", _binary_prices(p))
    return out


# ── track record (walk-forward) ──────────────────────────────────────────────

def _date_int(ts: Any) -> Optional[int]:
    if ts is None or pd.isna(ts):
        return None
    return int(pd.Timestamp(ts).strftime("%Y%m%d"))


def _build_track_record_for_league(league_key: str, history: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """(historical boards, walk-forward predictions) for every completed match of
    the league since RECORD_START_SEASON."""
    tour = tour_for_league(league_key)
    preds = wf.walk_forward(history, MODEL_PARAMS, start_season=RECORD_START_SEASON)
    if preds.empty:
        return [], preds
    d = wf.derive(preds["lam"].to_numpy(float), preds["mu"].to_numpy(float), preds["rho"].to_numpy(float), MAX_GOALS)
    preds = preds.assign(pH=d["pH"], pD=d["pD"], pA=d["pA"], league=league_key, tour=tour)
    boards: list[dict[str, Any]] = []
    for row in preds.itertuples(index=False):
        match_date = _date_int(row.date)
        home = getattr(row, "home_display", None) or _title(row.home)
        away = getattr(row, "away_display", None) or _title(row.away)
        board = build_soccer_backtest(
            tour=tour, home_team=home, away_team=away,
            home_goals=int(row.home_goals), away_goals=int(row.away_goals),
            prediction={"homeWin": row.pH, "draw": row.pD, "awayWin": row.pA},
            match_date=match_date, year=int(pd.Timestamp(row.date).year),
        )
        board.update({
            "season": season_label(int(row.season_start)),
            "lambdaHome": round(float(row.lam), 3),
            "lambdaAway": round(float(row.mu), 3),
            "modelVersion": MODEL_VERSION,
            "recordVersion": RECORD_VERSION,
        })
        boards.append(board)
    return boards, preds


def _compact_scores(block: dict[str, Any]) -> dict[str, Any]:
    """The track-record summary row for one (league|all, season) block."""
    def r(x: Any, nd: int = 4) -> Any:
        return round(float(x), nd) if isinstance(x, (int, float)) and np.isfinite(x) else None

    model = block.get("model", {})
    row: dict[str, Any] = {
        "n": model.get("n", 0),
        "model": {k: r(model.get(k)) for k in ("accuracy", "logLoss", "rps", "brier")},
        "alwaysHomeAccuracy": r(block.get("baselines", {}).get("alwaysHomeAccuracy")),
        "overUnder25": {
            "modelLogLoss": r(block.get("overUnder25", {}).get("logLoss")),
            "modelAccuracy": r(block.get("overUnder25", {}).get("accuracy")),
            "baseRateLogLoss": r(block.get("baselines", {}).get("over25BaseRateLogLoss")),
        },
        "btts": {
            "modelLogLoss": r(block.get("btts", {}).get("logLoss")),
            "baseRateLogLoss": r(block.get("baselines", {}).get("bttsBaseRateLogLoss")),
        },
        "correctScoreTop3HitRate": r(block.get("correctScore", {}).get("top3HitRate")),
        "naiveCorrectScoreTop3HitRate": r(block.get("correctScore", {}).get("naiveTop3HitRate")),
        "meanGoals": {"predicted": r(block.get("goals", {}).get("meanPredicted"), 3),
                      "actual": r(block.get("goals", {}).get("meanActual"), 3)},
        "market": None,
    }
    market = block.get("market")
    if market:
        row["market"] = {
            "n": market.get("n"),
            "favouriteWinRate": r(market.get("favouriteWinRate")),
            "logLoss": r(market.get("logLoss")),
            "rps": r(market.get("rps")),
            "source": market.get("source"),
        }
        if block.get("marketOverUnder25"):
            row["overUnder25"]["marketLogLoss"] = r(block["marketOverUnder25"].get("logLoss"))
    return row


def build_track_record_summary(preds: pd.DataFrame, world_cup_boards: list[dict[str, Any]]) -> dict[str, Any]:
    leagues: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    if not preds.empty:
        for (tour, season), g in preds.groupby(["tour", "season_start"], sort=True):
            leagues.append({"tour": tour, "season": season_label(int(season)), **_compact_scores(wf.score_predictions(g, MAX_GOALS))})
        for season, g in preds.groupby("season_start", sort=True):
            overall.append({"tour": "Top 5 leagues", "season": season_label(int(season)), **_compact_scores(wf.score_predictions(g, MAX_GOALS))})
    wc_hits = sum(1 for b in world_cup_boards if b.get("hitStatus") == "Top Pick")
    world_cup = {
        "tour": wc.TOUR_NAME,
        "season": "2026",
        "n": len(world_cup_boards),
        "topPickHits": wc_hits,
        "accuracy": round(wc_hits / len(world_cup_boards), 4) if world_cup_boards else None,
        "method": "Pre-tournament international Dixon-Coles model; archived record as published through "
                  "2026-07-12 plus later finished matches. Reported separately from the leagues.",
    }
    return {
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelVersion": MODEL_VERSION,
        "recordVersion": RECORD_VERSION,
        "method": (
            "Walk-forward: every match since the start of the "
            f"{season_label(RECORD_START_SEASON)} season is predicted by a Dixon-Coles model refitted twice a "
            "week (Monday and Friday) on matches played strictly before the refit, so no match is ever in its "
            "own training data. The market column is the football-data.co.uk closing average with the "
            "bookmaker margin removed, on the same matches."
        ),
        "leagues": leagues,
        "allLeagues": overall,
        "worldCup": world_cup,
        "disclaimer": DISCLAIMER,
    }


# ── upcoming fixtures ────────────────────────────────────────────────────────

def _published_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fit_current_model(history: pd.DataFrame, season: int) -> Optional[DixonColesModel]:
    train = wf.training_window(history, season, int(MODEL_PARAMS["windowSeasons"]))
    return wf.fit_model(train, MODEL_PARAMS)


def _fetch_league_fixtures(league_key: str, today) -> list[dict[str, Any]]:
    slug = str(LEAGUE_CONFIGS[league_key]["espn_slug"])
    window = f"{today - timedelta(days=RESULTS_DAYS_BACK):%Y%m%d}-{today + timedelta(days=UPCOMING_DAYS_AHEAD):%Y%m%d}"
    payload = client.fetch_espn_scoreboard(slug, dates=window)
    return client.parse_espn_fixtures(payload)


def _build_upcoming_for_league(league_key: str, model: DixonColesModel, fixtures: list[dict[str, Any]],
                               published_at: str) -> list[dict[str, Any]]:
    tour = tour_for_league(league_key)
    today_key = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    boards: list[dict[str, Any]] = []
    unknown: set[str] = set()
    for fx in fixtures:
        if fx.get("state") == "post":
            continue  # already finished
        if fx.get("date_int") and int(fx["date_int"]) < today_key:
            continue  # fetched only for results (e.g. a postponed fixture still "pre")
        home_name = fx.get("home_name") or ""
        away_name = fx.get("away_name") or ""
        home_key = resolve_team(home_name, model.teams)
        away_key = resolve_team(away_name, model.teams)
        for raw_name, key in ((home_name, home_key), (away_name, away_key)):
            if not model.knows_team(key):
                unknown.add(raw_name)
        lam, mu = model.rates(home_key, away_key)
        matrix = model.score_matrix(home_key, away_key, max_goals=MAX_GOALS)
        head = mk.score_matrix_markets(matrix)
        board_id = f"{league_key}-{fx.get('id')}"
        markets: list[dict[str, Any]] = []
        # Markets only before kick-off: a pre-match price on an in-play game misleads,
        # and the pick log would never grade it anyway.
        if fx.get("state") in (None, "pre"):
            markets = build_soccer_markets(board_id=board_id, home_label=home_name, away_label=away_name,
                                           matrix=matrix, published_at=published_at)
        boards.append(
            build_soccer_board(
                board_id=board_id,
                name=f"{home_name} vs {away_name}",
                tour=tour,
                home_team=home_name,
                away_team=away_name,
                home_win_probability=head["home"],
                draw_probability=head["draw"],
                away_win_probability=head["away"],
                scheduled_date=fx.get("date_int"),
                venue=fx.get("venue"),
                home_team_details=branding_from_espn_team(fx.get("home_team")),
                away_team_details=branding_from_espn_team(fx.get("away_team")),
                game_id=board_id,
                game_start=fx.get("start_iso"),
                lambda_home=lam,
                lambda_away=mu,
                markets=markets,
                basis="model",
                model_version=MODEL_VERSION,
            )
        )
    if unknown:
        logger.warning("soccer %s: teams unknown to the model (priced as league-average): %s",
                       league_key, ", ".join(sorted(unknown)))
    return boards


def _title(name: str) -> str:
    return name.title() if name and name.islower() else name


# ── World Cup ────────────────────────────────────────────────────────────────

def load_world_cup_archive(path: Path = WORLD_CUP_ARCHIVE) -> list[dict[str, Any]]:
    """The World Cup 2026 record exactly as it was published (archived 2026-07-12)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        logger.warning("World Cup archive unreadable (%s): %s", path, exc)
        return []
    return [b for b in data if isinstance(b, dict)]


def _build_world_cup_boards(today) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Return (upcoming WC match boards, completed WC backtest boards, title-odds board).

    The World Cup is just another ``tour`` ("FIFA World Cup") under the soccer
    key. Finished fixtures are scored against a pre-tournament model so each one
    records a fair out-of-sample hit/miss. The title simulation and the upcoming
    fixtures only run until the tournament ends.
    """
    window_start = datetime.strptime(WORLD_CUP_WINDOW[0], "%Y%m%d").date()
    window_end = datetime.strptime(WORLD_CUP_WINDOW[1], "%Y%m%d").date()
    if today < window_start - timedelta(days=60):
        return [], [], None
    tournament_live = today <= window_end
    try:
        results = client.fetch_international_results(since_year=2014)
        pre = results[results["date"].dt.strftime("%Y%m%d") < WORLD_CUP_WINDOW[0]]
        backtest_model = wc.build_international_model(pre if not pre.empty else results)
        model = wc.build_international_model(results) if tournament_live else backtest_model
    except Exception as exc:  # pragma: no cover - network / data
        logger.warning("World Cup model build failed: %s", exc)
        return [], [], None

    title_board: dict[str, Any] | None = None
    if tournament_live:
        try:
            groups = wc.load_world_cup_groups()
            probs = wc.simulate_tournament(model, groups, n_sims=WORLD_CUP_SIMS)
            ranked = sorted(probs.items(), key=lambda kv: kv[1]["win_title"], reverse=True)
            predictions = [
                {
                    "rank": i + 1,
                    "playerName": _title(team),
                    "winProbability": round(p["win_title"] * 100.0, 2),
                    "side": None,
                    "profile": {
                        "subtitle": f"Reach final {p['reach_final'] * 100:.0f}% · Advance {p['advance'] * 100:.0f}%",
                    },
                }
                for i, (team, p) in enumerate(ranked)
                if p["win_title"] > 0
            ]
            if predictions:
                title_board = {
                    "id": "fifa-world-cup-2026-title-odds",
                    "gameId": "fifa-world-cup-2026-title-odds",
                    "name": "World Cup 2026 — Title Odds",
                    "tour": wc.TOUR_NAME,
                    "course": "Monte Carlo tournament simulation",
                    "scheduledDate": int(WORLD_CUP_WINDOW[0]),
                    "latestDate": int(WORLD_CUP_WINDOW[0]),
                    "predictedWinner": predictions[0]["playerName"],
                    "predictions": predictions,
                    "markets": [],
                }
        except Exception as exc:  # pragma: no cover
            logger.warning("World Cup simulation failed: %s", exc)

    full_history = None
    team_index: dict[str, str] = {}
    if tournament_live:
        try:
            full_history = client.fetch_international_results(since_year=1990)
        except Exception:  # pragma: no cover
            full_history = None
        try:
            team_index = client.fetch_espn_team_index("fifa.world")
        except Exception:  # pragma: no cover
            team_index = {}
    roster_cache: dict[str, list[dict[str, Any]]] = {}

    def _roster_for(team_name: str) -> list[dict[str, Any]]:
        key = canonical_national_name(team_name)
        if key in roster_cache:
            return roster_cache[key]
        tid = team_index.get(key)
        players = client.fetch_espn_roster("fifa.world", tid) if tid else []
        roster_cache[key] = players
        return players

    match_boards: list[dict[str, Any]] = []
    completed_boards: list[dict[str, Any]] = []
    try:
        payload = client.fetch_espn_scoreboard("fifa.world", dates=f"{WORLD_CUP_WINDOW[0]}-{WORLD_CUP_WINDOW[1]}")
        fixtures = client.parse_espn_fixtures(payload)

        finished = [
            f for f in fixtures
            if f.get("state") == "post" and f.get("completed")
            and f.get("home_score") is not None and f.get("away_score") is not None
        ]
        finished.sort(key=lambda f: f.get("date_int") or 0, reverse=True)
        for fx in finished:
            home_name = fx.get("home_name") or ""
            away_name = fx.get("away_name") or ""
            if not home_name or not away_name:
                continue
            try:
                neutral = fx.get("neutral_site") is not False
                pred = backtest_model.predict_match(
                    canonical_national_name(home_name), canonical_national_name(away_name), neutral=neutral
                )
                date_int = fx.get("date_int")
                year = int(str(date_int)[:4]) if date_int else today.year
                board = build_soccer_backtest(
                    tour=wc.TOUR_NAME,
                    home_team=home_name,
                    away_team=away_name,
                    home_goals=int(fx["home_score"]),
                    away_goals=int(fx["away_score"]),
                    prediction=pred,
                    match_date=date_int,
                    year=year,
                )
                board["recordVersion"] = f"worldcup-pretournament:{MODEL_VERSION}"
                completed_boards.append(board)
            except Exception as exc:  # pragma: no cover
                logger.warning("WC backtest board failed for %s v %s: %s", home_name, away_name, exc)

        if tournament_live:
            upcoming_fixtures = [f for f in fixtures if f.get("state") != "post"]
            upcoming_fixtures.sort(key=lambda f: f.get("date_int") or 99999999)
            for fx in upcoming_fixtures[:WORLD_CUP_MAX_FIXTURES]:
                home_name = fx.get("home_name") or ""
                away_name = fx.get("away_name") or ""
                neutral = fx.get("neutral_site") is not False
                pred = model.predict_match(
                    canonical_national_name(home_name), canonical_national_name(away_name), neutral=neutral
                )
                board = build_soccer_board(
                    board_id=f"fifa.world-{fx.get('id')}",
                    name=f"{home_name} vs {away_name}",
                    tour=wc.TOUR_NAME,
                    home_team=home_name,
                    away_team=away_name,
                    home_win_probability=pred["homeWin"],
                    draw_probability=pred["draw"],
                    away_win_probability=pred["awayWin"],
                    scheduled_date=fx.get("date_int"),
                    venue=fx.get("venue"),
                    home_team_details=branding_from_espn_team(fx.get("home_team")),
                    away_team_details=branding_from_espn_team(fx.get("away_team")),
                    game_start=fx.get("start_iso"),
                    lambda_home=pred["lambdaHome"],
                    lambda_away=pred["lambdaAway"],
                    # International markets are not validated (no totals/BTTS backtest
                    # for the national-team model), so World Cup boards carry none.
                    markets=[],
                    basis="model",
                    model_version=f"worldcup-{MODEL_VERSION}",
                )
                try:
                    if full_history is not None and not full_history.empty:
                        board["headToHead"] = wcd.head_to_head(full_history, home_name, away_name)
                        board["teamHistory"] = [
                            wcd.team_history(full_history, home_name),
                            wcd.team_history(full_history, away_name),
                        ]
                except Exception as exc:  # pragma: no cover
                    logger.warning("WC detail (h2h/history) failed for %s v %s: %s", home_name, away_name, exc)
                try:
                    rosters = []
                    for tname in (home_name, away_name):
                        players = _roster_for(tname)
                        if players:
                            rosters.append({"team": tname, "players": players})
                    if rosters:
                        board["rosters"] = rosters
                except Exception as exc:  # pragma: no cover
                    logger.warning("WC roster fetch failed for %s v %s: %s", home_name, away_name, exc)
                match_boards.append(board)
    except Exception as exc:  # pragma: no cover
        logger.warning("World Cup fixtures fetch failed: %s", exc)

    return match_boards, completed_boards, title_board


# ── history merge (ledger: once published, a graded board stays published) ───

def _board_key(board: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(board.get("tour") or "").upper(),
        " ".join(str(board.get("tournament") or board.get("name") or "").lower().split()),
        int(board.get("scheduledDate") or board.get("latestDate") or 0),
    )


def merge_history(new_boards: list[dict[str, Any]], *, archive: list[dict[str, Any]],
                  previous: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union of the regenerated record, the World Cup archive and the previously
    exported file, newest first, one board per (tour, match, date).

    * World Cup boards: archive first (as published), then the previous file, then
      newly scored matches -- an already-published WC board is never re-scored.
    * League boards: the regenerated walk-forward wins; boards from the previous
      file are kept only if they carry the same recordVersion (so a source hiccup
      cannot shrink the record), while legacy static-fit boards are dropped once.
    """
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    wc_upper = wc.TOUR_NAME.upper()
    for board in archive:
        merged.setdefault(_board_key(board), board)
    for board in previous:
        key = _board_key(board)
        if key[0] == wc_upper:
            merged.setdefault(key, board)
    for board in new_boards:
        key = _board_key(board)
        if key[0] == wc_upper:
            merged.setdefault(key, board)
        else:
            merged[key] = board
    for board in previous:
        key = _board_key(board)
        if key[0] != wc_upper and board.get("recordVersion") == RECORD_VERSION:
            merged.setdefault(key, board)
    return sorted(merged.values(), key=lambda b: (int(b.get("latestDate") or b.get("scheduledDate") or 0),
                                                   str(b.get("tournament") or "")), reverse=True)


# ── market log grading ───────────────────────────────────────────────────────

def _league_of(board_id: str) -> str:
    return str(board_id or "").split("-", 1)[0]


def collect_results(picks: dict[str, dict[str, Any]], espn_results: dict[str, tuple[int, int]],
                    histories: dict[str, pd.DataFrame]) -> dict[str, tuple[int, int]]:
    """gameId -> (home, away) final score for logged picks. ESPN finished fixtures
    first (same id as the board); otherwise the football-data result of the same
    fixture (league, both teams, date within 3 days) so older picks stay gradable
    without re-fetching months of ESPN scoreboards."""
    results: dict[str, tuple[int, int]] = {}
    for pick in picks.values():
        gid = str(pick.get("gameId") or "")
        if not gid or gid in results:
            continue
        if gid in espn_results:
            results[gid] = espn_results[gid]
            continue
        hist = histories.get(_league_of(pick.get("boardId") or gid))
        game_date = pick.get("gameDate")
        if hist is None or hist.empty or not game_date:
            continue
        known = set(hist["home"]) | set(hist["away"])
        home = resolve_team(pick.get("homeTeam"), known)
        away = resolve_team(pick.get("awayTeam"), known)
        try:
            day = pd.Timestamp(datetime.strptime(str(int(game_date)), "%Y%m%d"))
        except ValueError:
            continue
        rows = hist[(hist["home"] == home) & (hist["away"] == away) & ((hist["date"] - day).abs() <= pd.Timedelta(days=3))]
        if len(rows) == 1:
            results[gid] = (int(rows["home_goals"].iloc[0]), int(rows["away_goals"].iloc[0]))
    return results


def _market_picks_root() -> Path:
    override = os.getenv(MARKET_PICKS_ROOT_ENV)
    return Path(override) if override else market_log.DEFAULT_ROOT


def log_and_grade_markets(upcoming: list[dict[str, Any]], espn_results: dict[str, tuple[int, int]],
                          histories: dict[str, pd.DataFrame], *, published_at: str,
                          root: Optional[Path] = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Append this bake's picks to the log, then grade everything logged so far.
    Returns (graded history newest first, capped; summary rows)."""
    root = root or _market_picks_root()
    records = market_log.picks_from_boards(SPORT_KEY, upcoming, season_of=SEASON_OF,
                                           model_version=MODEL_VERSION, published_at=published_at)
    market_log.write_snapshot(SPORT_KEY, records, root=root)
    chosen = market_log.pregame_picks(market_log.load_snapshots(SPORT_KEY, root=root))
    results = collect_results(chosen, espn_results, histories)
    graded = market_log.grade_picks(chosen, results)
    graded.sort(key=lambda g: (str(g.get("gameStart") or ""), str(g.get("marketId") or "")), reverse=True)
    summary = market_log.summarize(graded)
    return graded[:MARKET_HISTORY_CAP], summary


# ── payload ──────────────────────────────────────────────────────────────────

def _empty_payload(selected_date: str | None) -> dict[str, Any]:
    return {
        "selectedDate": selected_date,
        "availableDates": [],
        "upcoming": [],
        "completed": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": LIVE_SOURCE,
    }


def _history_seasons(season: int) -> int:
    """Seasons to load: the record's first season and its 5-season window, up to now."""
    window = int(MODEL_PARAMS["windowSeasons"])
    return max(window + 1, season - RECORD_START_SEASON + window)


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    """Return the soccer board payload across all predictable leagues (JSON-able;
    used by the divination /api/sports/soccer/boards fallback). Nothing is logged."""
    payload, _, _ = _build_payload(selected_date)
    return payload


def _build_payload(selected_date: str | None = None, previous_history: Optional[list[dict[str, Any]]] = None,
                   published_at: Optional[str] = None
                   ) -> tuple[dict[str, Any], dict[str, tuple[int, int]], dict[str, pd.DataFrame]]:
    """(payload, ESPN final scores by gameId, league histories). The last two feed
    the market-log grading in export_soccer_frontend_data."""
    published_at = published_at or _published_now()
    today = datetime.now(timezone.utc).date()
    season = current_season_start(today)
    upcoming: list[dict[str, Any]] = []
    record_boards: list[dict[str, Any]] = []
    record_preds: list[pd.DataFrame] = []
    espn_results: dict[str, tuple[int, int]] = {}
    histories: dict[str, pd.DataFrame] = {}

    for league_key in PREDICTABLE_LEAGUE_KEYS:
        try:
            history = client.load_history(league_key, seasons=_history_seasons(season),
                                          current_season_start=season, keep_odds=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("soccer history load failed for %s: %s", league_key, exc)
            continue
        if history.empty:
            continue
        histories[league_key] = history

        try:
            boards, preds = _build_track_record_for_league(league_key, history)
            record_boards.extend(boards)
            if not preds.empty:
                record_preds.append(preds)
        except Exception as exc:  # pragma: no cover
            logger.warning("soccer track record failed for %s: %s", league_key, exc)

        try:
            model = _fit_current_model(history, season)
        except Exception as exc:  # pragma: no cover
            logger.warning("Dixon-Coles fit failed for %s: %s", league_key, exc)
            model = None
        if model is None:
            continue
        try:
            fixtures = _fetch_league_fixtures(league_key, today)
        except Exception as exc:  # pragma: no cover - network
            logger.warning("soccer fixtures fetch failed for %s: %s", league_key, exc)
            fixtures = []
        for fx in fixtures:
            if fx.get("state") == "post" and fx.get("completed") and fx.get("home_score") is not None \
                    and fx.get("away_score") is not None:
                espn_results[f"{league_key}-{fx.get('id')}"] = (int(fx["home_score"]), int(fx["away_score"]))
        upcoming.extend(_build_upcoming_for_league(league_key, model, fixtures, published_at))

    world_cup_completed: list[dict[str, Any]] = []
    if WORLD_CUP_ENABLED:
        try:
            wc_matches, world_cup_completed, title_board = _build_world_cup_boards(today)
            if title_board:
                upcoming.append(title_board)
            upcoming.extend(wc_matches)
        except Exception as exc:  # pragma: no cover
            logger.warning("World Cup board build failed: %s", exc)

    completed = merge_history(record_boards + world_cup_completed, archive=load_world_cup_archive(),
                              previous=previous_history or [])
    upcoming.sort(key=lambda b: (b.get("scheduledDate") or 99999999, str(b.get("gameStart") or "")))

    preds_all = pd.concat(record_preds, ignore_index=True) if record_preds else pd.DataFrame()
    wc_boards = [b for b in completed if str(b.get("tour") or "") == wc.TOUR_NAME]
    payload = {
        "selectedDate": selected_date,
        "availableDates": [],
        "upcoming": upcoming,
        "completed": completed,
        "trackRecord": build_track_record_summary(preds_all, wc_boards),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": LIVE_SOURCE,
    }
    return payload, espn_results, histories


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [b for b in data if isinstance(b, dict)] if isinstance(data, list) else []


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def export_soccer_frontend_data() -> None:
    """Write the precomputed static soccer JSON the BFF serves (like tennis/golf),
    append this bake's market picks to the log and grade the log."""
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    history_path = FRONTEND_DATA_DIR / "soccer_historical_backtests.json"
    published_at = _published_now()

    payload, espn_results, histories = _build_payload(previous_history=_read_json_list(history_path),
                                                      published_at=published_at)
    upcoming = payload.get("upcoming", [])
    completed = payload.get("completed", [])

    graded, summary = [], []
    try:
        graded, summary = log_and_grade_markets(upcoming, espn_results, histories, published_at=published_at)
    except Exception as exc:  # pragma: no cover - the boards still ship without the log
        logger.warning("soccer market log/grade failed: %s", exc)

    _write_json(FRONTEND_DATA_DIR / "soccer_upcoming_tournaments.json", upcoming)
    _write_json(history_path, completed)
    _write_json(FRONTEND_DATA_DIR / "soccer_track_record.json", payload.get("trackRecord", {}))
    _write_json(FRONTEND_DATA_DIR / f"{SPORT_KEY}_market_history.json", graded)
    _write_json(FRONTEND_DATA_DIR / f"{SPORT_KEY}_market_summary.json", summary)

    n_markets = sum(len(b.get("markets") or []) for b in upcoming)
    print(f"Exported {len(upcoming)} soccer upcoming boards ({n_markets} market picks)")
    print(f"Exported {len(completed)} soccer historical boards")
    print(f"Graded {len(graded)} soccer market picks; {len(summary)} summary rows")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export_soccer_frontend_data()
