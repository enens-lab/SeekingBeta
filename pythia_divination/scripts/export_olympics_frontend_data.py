"""Summer Olympics frontend data export.

Produces a ranked "field" board projecting the next Summer Games medal table
(countries ranked by projected medals), plus a held-out backtest board, under
the ``olympics`` sport key. Renders through the generic ranked renderer, so no
sport-specific UI is required.

NOTE: the bundled historical dataset ends at Rio 2016, so the projection is the
model rolled forward from the most recent available editions. It is a medal
*forecast*, deliberately scoped to the tractable Olympic prediction problem
(country medal counts) rather than individual timed/judged events.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DIV_ROOT = Path(__file__).resolve().parents[1]
if str(DIV_ROOT) not in sys.path:
    sys.path.insert(0, str(DIV_ROOT))

PROJ_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DATA_DIR = PROJ_ROOT / "pythia_prophecy" / "frontend" / "src" / "data"

from sports.olympics import client, medal_model as mm, detail as od
from sports.olympics.constants import OLYMPIC_EDITIONS, noc_display_name

logger = logging.getLogger(__name__)

LIVE_SOURCE = "divination_olympics_medal_model"
# Number of past Summer Games to expose as a historical browser (most recent N).
HISTORICAL_GAMES_COUNT = 4


def _empty_payload(selected_date: str | None) -> dict[str, Any]:
    return {
        "selectedDate": selected_date,
        "availableDates": [],
        "upcoming": [],
        "completed": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": LIVE_SOURCE,
    }


def _hosts_for(games: dict) -> dict[int, str]:
    return mm.WINTER_HOSTS if games["season"] == "Winter" else mm.SUMMER_HOSTS


def _projection_board(table, games: dict) -> dict[str, Any] | None:
    proj = mm.project_next_games(table, host_noc=games["host_noc"], hosts=_hosts_for(games))
    if proj.empty:
        return None
    proj = proj.head(40)
    total = float(proj["projected"].sum()) or 1.0
    predictions = []
    for i, r in enumerate(proj.itertuples(index=False)):
        predictions.append(
            {
                "rank": i + 1,
                "playerName": noc_display_name(r.noc),
                "winProbability": round(r.projected / total * 100.0, 2),
                "side": None,
                "profile": {"subtitle": f"~{round(r.projected)} medals projected"},
            }
        )
    return {
        "id": f"olympics-{games['season'].lower()}-{games['year']}-medal-table",
        "name": f"{games['city']} {games['year']} — Projected Medal Table",
        "tour": games["tour"],
        "course": "Gradient-boosted medal-count model",
        "scheduledDate": int(games.get("start_date") or f"{games['year']}0714"),
        "latestDate": int(games.get("end_date") or f"{games['year']}0714"),
        "predictedWinner": predictions[0]["playerName"],
        "predictions": predictions,
    }


def _backtest_board(table, games: dict) -> dict[str, Any] | None:
    bt = mm.backtest_last_games(table, hosts=_hosts_for(games))
    rows = bt.get("rows") or []
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda x: x["predicted"], reverse=True)
    actual_top = max(rows, key=lambda x: x["actual"])
    pred_top = rows_sorted[0]
    total_pred = sum(r["predicted"] for r in rows_sorted) or 1.0

    full_field = []
    for i, r in enumerate(rows_sorted[:40]):
        full_field.append(
            {
                "rank": i + 1,
                "playerName": noc_display_name(r["noc"]),
                "winProbability": round(r["predicted"] / total_pred * 100.0, 2),
                "actualWinner": r["noc"] == actual_top["noc"],
                "side": None,
                "profile": {"subtitle": f"predicted ~{round(r['predicted'])} · actual {round(r['actual'])} medals"},
            }
        )

    hit = pred_top["noc"] == actual_top["noc"]
    return {
        "year": int(bt["year"]),
        "tournament": f"{bt['year']} {games['season']} Olympics — Medal Table (backtest)",
        "tour": games["tour"],
        "hitStatus": "Top Pick" if hit else "Miss",
        "predictedWinner": noc_display_name(pred_top["noc"]),
        "actualWinner": f"{noc_display_name(actual_top['noc'])} ({round(actual_top['actual'])} medals)",
        "prob": round(pred_top["predicted"] / total_pred, 4),
        "scheduledDate": int(f"{bt['year']}0801"),
        "latestDate": int(f"{bt['year']}0801"),
        "fullField": full_field,
    }


def _discipline_boards_for_year(df, year: int, games: dict, *, historical: bool) -> list[dict[str, Any]]:
    """One board per sport for a Games year (season-aware), carrying its
    disciplines+medalists. Renders as expandable cards: sport -> events ->
    gold/silver/bronze. Used for the latest Games (upcoming) + past Games
    (historical browser). Tagged with the season's tour."""
    season = games["season"]
    boards: list[dict[str, Any]] = []
    for sport in od.sports_for_year(df, year, season):
        disciplines = od.disciplines_for_sport(df, sport, year, season)
        if not disciplines:
            continue
        n_events = len(disciplines)
        n_medalists = sum(len(d["medalists"]) for d in disciplines)
        board: dict[str, Any] = {
            "id": f"olympics-{season.lower()}-{year}-{sport.lower().replace(' ', '-')}",
            "name": f"{sport}",
            "tour": games["tour"],
            "course": f"{n_events} events · {n_medalists} medalists",
            "scheduledDate": int(f"{year}0714"),
            "latestDate": int(f"{year}0730"),
            "disciplines": disciplines,
            "predictions": [],
        }
        if historical:
            board.update({
                "year": int(year),
                "tournament": f"{sport} — {year} {season} Olympics",
                "hitStatus": "Top Pick",  # neutral label; these are results, not predictions
                "predictedWinner": None,
            })
        boards.append(board)
    return boards


def _build_edition(df, games: dict) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All boards for one Olympic edition (Summer or Winter): projection +
    latest-Games disciplines (upcoming) and backtest + historical browser
    (completed)."""
    season = games["season"]
    upcoming: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    # An edition is "upcoming" only until its Games end; a finished edition (e.g. the
    # 2026 Winter Games once they're over) must never linger under Upcoming as a stale
    # projection. Defaults to end-of-year if an edition somehow lacks an end_date.
    today_yyyymmdd = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    edition_upcoming = int(games.get("end_date") or f"{games['year']}1231") >= today_yyyymmdd
    try:
        table = client.medal_table(df, season=season)
    except Exception as exc:  # pragma: no cover
        logger.warning("%s medal table failed: %s", season, exc)
        return upcoming, completed

    try:
        board = _projection_board(table, games)
        if board and edition_upcoming:
            upcoming.append(board)
    except Exception as exc:  # pragma: no cover
        logger.warning("%s projection failed: %s", season, exc)
    try:
        bt = _backtest_board(table, games)
        if bt:
            completed.append(bt)
    except Exception as exc:  # pragma: no cover
        logger.warning("%s backtest failed: %s", season, exc)

    try:
        years = sorted(df[df["Season"] == season]["Year"].dropna().unique().tolist(), reverse=True)
        latest = int(years[0]) if years else None
        if latest is not None:
            upcoming.extend(_discipline_boards_for_year(df, latest, games, historical=False))
        for year in [int(y) for y in years[:HISTORICAL_GAMES_COUNT]]:
            completed.extend(_discipline_boards_for_year(df, year, games, historical=True))
    except Exception as exc:  # pragma: no cover
        logger.warning("%s discipline detail failed: %s", season, exc)

    return upcoming, completed


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    """Olympics board payload: both Summer (2028) and Winter (2026) editions as
    separate tours under the one olympics key."""
    try:
        df = client.load_athlete_events()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("Olympics dataset load failed: %s", exc)
        return _empty_payload(selected_date)

    upcoming: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for games in OLYMPIC_EDITIONS:
        ed_up, ed_comp = _build_edition(df, games)
        upcoming.extend(ed_up)
        completed.extend(ed_comp)

    return {
        "selectedDate": selected_date,
        "availableDates": [],
        "upcoming": upcoming,
        "completed": completed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": LIVE_SOURCE,
    }


def _json_default(o: Any) -> Any:
    # numpy scalars/arrays aren't JSON-serializable; convert to native so board
    # numbers stay NUMBERS (not stringified) for the frontend. Fall back to str.
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def export_olympics_frontend_data() -> None:
    import json

    payload = build_live_upcoming_payload()
    upcoming = payload["upcoming"]
    completed = payload["completed"]
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA_DIR / "olympics_upcoming_tournaments.json").write_text(
        json.dumps(upcoming, indent=2, default=_json_default)
    )
    (FRONTEND_DATA_DIR / "olympics_historical_backtests.json").write_text(
        json.dumps(completed, indent=2, default=_json_default)
    )
    print(f"Exported {len(upcoming)} olympics upcoming boards")
    print(f"Exported {len(completed)} olympics historical boards")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export_olympics_frontend_data()
