"""Soccer (association football) frontend data export.

Builds the live soccer board payload consumed by the divination
``/api/sports/soccer/boards`` endpoint and proxied by the prophecy BFF.

Soccer is ONE sport key whose ``tour`` carries the competition (Premier
League, La Liga, Serie A, Bundesliga, Ligue 1, ...), mirroring golf/tennis.
Match outcomes are 1X2 (home / draw / away) from a Dixon-Coles model.

Pipeline per league:
  1. load multi-season match history (football-data.co.uk)
  2. OUT-OF-SAMPLE backtests: fit on matches before a cutoff, predict the
     recent held-out matches -> Track Record (honest, not in-sample)
  3. fit the full time-decayed model and predict upcoming ESPN fixtures -> 1X2

DRAW / 1X2 CONTRACT: each board carries homeWinProbability / drawProbability /
awayWinProbability (Optional in the shared contract) plus a ``predictions``
mirror of [Home, Draw, Away] rows so older clients still render something.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

from sports.soccer import client
from sports.soccer.branding import branding_from_espn_team
from sports.soccer.constants import (
    DEFAULT_HISTORY_SEASONS,
    PREDICTABLE_LEAGUE_KEYS,
    LEAGUE_CONFIGS,
    canonical_team_name,
    tour_for_league,
)
from sports.soccer.dixon_coles import DixonColesModel
from sports.soccer import world_cup as wc

logger = logging.getLogger(__name__)

LIVE_SOURCE = "divination_live_soccer_feed"
UPCOMING_DAYS_AHEAD = 21
BACKTEST_HOLDOUT_PER_LEAGUE = 40
MAX_COMPLETED_BOARDS = 90
TIME_DECAY_XI = 0.0018  # per day; ~1-year half life

WORLD_CUP_ENABLED = True
WORLD_CUP_SIMS = 10000
WORLD_CUP_MAX_FIXTURES = 32  # cap upcoming WC match boards in the feed
WORLD_CUP_WINDOW = ("20260611", "20260719")  # tournament dates


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
    }
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
    }


def _decay_weights(dates) -> np.ndarray:
    valid = dates.dropna()
    if valid.empty:
        return np.ones(len(dates))
    latest = valid.max()
    days_ago = (latest - dates).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
    return np.exp(-TIME_DECAY_XI * days_ago)


def _fit_model(history) -> DixonColesModel:
    weights = _decay_weights(history["date"])
    return DixonColesModel().fit(
        history["home"], history["away"],
        history["home_goals"].astype(int), history["away_goals"].astype(int),
        weights=weights,
    )


def _build_backtests_for_league(league_key: str, history) -> list[dict[str, Any]]:
    tour = tour_for_league(league_key)
    if len(history) < 80:
        return []
    history = history.sort_values("date").reset_index(drop=True)
    # Train on the older portion, evaluate out-of-sample on the recent third
    # (~one season) rather than only the noisy final gameweeks.
    holdout = max(BACKTEST_HOLDOUT_PER_LEAGUE, len(history) // 3)
    holdout = min(holdout, len(history) - 50)
    if holdout <= 0:
        return []
    train = history.iloc[:-holdout]
    test = history.iloc[-holdout:]
    if len(train) < 50 or test.empty:
        return []

    model = _fit_model(train)
    boards: list[dict[str, Any]] = []
    for row in test.itertuples(index=False):
        pred = model.predict_match(row.home, row.away)
        match_date = int(row.date.strftime("%Y%m%d")) if row.date is not None and not _is_nat(row.date) else None
        year = int(row.date.year) if match_date else datetime.now(timezone.utc).year
        boards.append(
            build_soccer_backtest(
                tour=tour,
                home_team=_title(row.home),
                away_team=_title(row.away),
                home_goals=int(row.home_goals),
                away_goals=int(row.away_goals),
                prediction=pred,
                match_date=match_date,
                year=year,
            )
        )
    return boards


def _build_upcoming_for_league(league_key: str, model: DixonColesModel) -> list[dict[str, Any]]:
    cfg = LEAGUE_CONFIGS[league_key]
    tour = str(cfg["tour"])
    slug = str(cfg["espn_slug"])
    today = datetime.now(timezone.utc).date()
    window = f"{today:%Y%m%d}-{today + timedelta(days=UPCOMING_DAYS_AHEAD):%Y%m%d}"

    try:
        payload = client.fetch_espn_scoreboard(slug, dates=window)
        fixtures = client.parse_espn_fixtures(payload)
    except Exception as exc:  # pragma: no cover - network
        logger.warning("soccer upcoming fetch failed for %s: %s", league_key, exc)
        return []

    boards: list[dict[str, Any]] = []
    for fx in fixtures:
        if fx.get("state") == "post":
            continue  # already finished
        home_name = fx.get("home_name") or ""
        away_name = fx.get("away_name") or ""
        pred = model.predict_match(canonical_team_name(home_name), canonical_team_name(away_name))
        boards.append(
            build_soccer_board(
                board_id=f"{league_key}-{fx.get('id')}",
                name=f"{home_name} vs {away_name}",
                tour=tour,
                home_team=home_name,
                away_team=away_name,
                home_win_probability=pred["homeWin"],
                draw_probability=pred["draw"],
                away_win_probability=pred["awayWin"],
                scheduled_date=fx.get("date_int"),
                venue=fx.get("venue"),
                home_team_details=branding_from_espn_team(fx.get("home_team")),
                away_team_details=branding_from_espn_team(fx.get("away_team")),
            )
        )
    return boards


def _is_nat(value: Any) -> bool:
    try:
        return value != value  # NaT != NaT
    except Exception:  # pragma: no cover
        return False


def _title(name: str) -> str:
    return name.title() if name and name.islower() else name


def _build_world_cup_boards() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (upcoming WC match boards, title-odds ranked board).

    The World Cup is just another ``tour`` ("FIFA World Cup") under the soccer
    key, so these boards flow through the existing pipeline unchanged.
    """
    try:
        groups = wc.load_world_cup_groups()
        results = client.fetch_international_results(since_year=2014)
        model = wc.build_international_model(results)
    except Exception as exc:  # pragma: no cover - network / data
        logger.warning("World Cup model build failed: %s", exc)
        return [], None

    # --- title-odds ranked "field" board (renders like a golf field) ---
    title_board: dict[str, Any] | None = None
    try:
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
                "name": "World Cup 2026 — Title Odds",
                "tour": wc.TOUR_NAME,
                "course": "Monte Carlo tournament simulation",
                "scheduledDate": int(WORLD_CUP_WINDOW[0]),
                "latestDate": int(WORLD_CUP_WINDOW[0]),
                "predictedWinner": predictions[0]["playerName"],
                "predictions": predictions,
            }
    except Exception as exc:  # pragma: no cover
        logger.warning("World Cup simulation failed: %s", exc)

    # --- upcoming WC fixtures as 1X2 boards (neutral) ---
    match_boards: list[dict[str, Any]] = []
    try:
        payload = client.fetch_espn_scoreboard("fifa.world", dates=f"{WORLD_CUP_WINDOW[0]}-{WORLD_CUP_WINDOW[1]}")
        fixtures = client.parse_espn_fixtures(payload)
        fixtures = [f for f in fixtures if f.get("state") != "post"]
        fixtures.sort(key=lambda f: f.get("date_int") or 99999999)
        for fx in fixtures[:WORLD_CUP_MAX_FIXTURES]:
            home_name = fx.get("home_name") or ""
            away_name = fx.get("away_name") or ""
            from sports.soccer.constants import canonical_national_name
            pred = model.predict_match(
                canonical_national_name(home_name), canonical_national_name(away_name), neutral=True
            )
            match_boards.append(
                build_soccer_board(
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
                )
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("World Cup fixtures fetch failed: %s", exc)

    return match_boards, title_board


def _empty_payload(selected_date: str | None) -> dict[str, Any]:
    return {
        "selectedDate": selected_date,
        "availableDates": [],
        "upcoming": [],
        "completed": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": LIVE_SOURCE,
    }


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    """Return the live soccer board payload across all predictable leagues."""
    upcoming: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []

    for league_key in PREDICTABLE_LEAGUE_KEYS:
        try:
            history = client.load_history(league_key, seasons=DEFAULT_HISTORY_SEASONS)
        except Exception as exc:  # pragma: no cover
            logger.warning("soccer history load failed for %s: %s", league_key, exc)
            continue
        if history.empty:
            continue

        completed.extend(_build_backtests_for_league(league_key, history))
        try:
            model = _fit_model(history)
        except Exception as exc:  # pragma: no cover
            logger.warning("Dixon-Coles fit failed for %s: %s", league_key, exc)
            continue
        upcoming.extend(_build_upcoming_for_league(league_key, model))

    # FIFA World Cup (international model + Monte Carlo) — just another tour.
    if WORLD_CUP_ENABLED:
        try:
            wc_matches, title_board = _build_world_cup_boards()
            if title_board:
                upcoming.append(title_board)
            upcoming.extend(wc_matches)
        except Exception as exc:  # pragma: no cover
            logger.warning("World Cup board build failed: %s", exc)

    upcoming.sort(key=lambda b: (b.get("scheduledDate") or 99999999))
    completed.sort(key=lambda b: (b.get("latestDate") or 0), reverse=True)
    completed = completed[:MAX_COMPLETED_BOARDS]

    return {
        "selectedDate": selected_date,
        "availableDates": [],
        "upcoming": upcoming,
        "completed": completed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": LIVE_SOURCE,
    }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    payload = build_live_upcoming_payload()
    print(json.dumps({
        "upcoming": len(payload["upcoming"]),
        "completed": len(payload["completed"]),
        "source": payload["source"],
        "sample_completed": payload["completed"][:2],
        "sample_upcoming": payload["upcoming"][:2],
    }, indent=2, default=str))
