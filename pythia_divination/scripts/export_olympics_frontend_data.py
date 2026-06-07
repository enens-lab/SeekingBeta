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

from sports.olympics import client, medal_model as mm, detail as od
from sports.olympics.constants import NEXT_SUMMER_GAMES, TOUR_NAME, noc_display_name

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


def _projection_board(table) -> dict[str, Any] | None:
    proj = mm.project_next_games(table, host_noc=NEXT_SUMMER_GAMES["host_noc"])
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
    games = NEXT_SUMMER_GAMES
    return {
        "id": f"olympics-{games['year']}-medal-table",
        "name": f"{games['city']} {games['year']} — Projected Medal Table",
        "tour": TOUR_NAME,
        "course": "Gradient-boosted medal-count model",
        "scheduledDate": int(f"{games['year']}0714"),
        "latestDate": int(f"{games['year']}0714"),
        "predictedWinner": predictions[0]["playerName"],
        "predictions": predictions,
    }


def _backtest_board(table) -> dict[str, Any] | None:
    bt = mm.backtest_last_games(table)
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
        "tournament": f"{bt['year']} Summer Olympics — Medal Table (backtest)",
        "tour": TOUR_NAME,
        "hitStatus": "Top Pick" if hit else "Miss",
        "predictedWinner": noc_display_name(pred_top["noc"]),
        "actualWinner": f"{noc_display_name(actual_top['noc'])} ({round(actual_top['actual'])} medals)",
        "prob": round(pred_top["predicted"] / total_pred, 4),
        "scheduledDate": int(f"{bt['year']}0801"),
        "latestDate": int(f"{bt['year']}0801"),
        "fullField": full_field,
    }


def _discipline_boards_for_year(df, year: int, *, historical: bool) -> list[dict[str, Any]]:
    """One board per sport for a Games year, carrying its disciplines+medalists.

    These render as expandable cards: sport -> events -> gold/silver/bronze. Used
    for both the latest Games (upcoming-side detail) and past Games (historical
    browser). Each board's `disciplines` list is the events under that sport."""
    boards: list[dict[str, Any]] = []
    for sport in od.sports_for_year(df, year):
        disciplines = od.disciplines_for_sport(df, sport, year)
        if not disciplines:
            continue
        n_events = len(disciplines)
        n_medalists = sum(len(d["medalists"]) for d in disciplines)
        board: dict[str, Any] = {
            "id": f"olympics-{year}-{sport.lower().replace(' ', '-')}",
            "name": f"{sport}",
            "tour": TOUR_NAME,
            "course": f"{n_events} events · {n_medalists} medalists",
            "scheduledDate": int(f"{year}0714"),
            "latestDate": int(f"{year}0730"),
            "disciplines": disciplines,
            "predictions": [],
        }
        if historical:
            board.update({
                "year": int(year),
                "tournament": f"{sport} — {year} Summer Olympics",
                "hitStatus": "Top Pick",  # neutral label; these are results, not predictions
                "predictedWinner": None,
            })
        boards.append(board)
    return boards


def build_live_upcoming_payload(selected_date: str | None = None) -> dict[str, Any]:
    """Return the Olympics medal-table board payload."""
    try:
        table = client.summer_medal_table()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("Olympics medal table load failed: %s", exc)
        return _empty_payload(selected_date)

    upcoming: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    try:
        board = _projection_board(table)
        if board:
            upcoming.append(board)
    except Exception as exc:  # pragma: no cover
        logger.warning("Olympics projection failed: %s", exc)
    try:
        bt = _backtest_board(table)
        if bt:
            completed.append(bt)
    except Exception as exc:  # pragma: no cover
        logger.warning("Olympics backtest failed: %s", exc)

    # Discipline drill-down (sport -> events -> medalists) from the athlete dataset.
    try:
        df = client.load_athlete_events()
        years = sorted(df[df["Season"] == "Summer"]["Year"].dropna().unique().tolist(), reverse=True)
        latest = int(years[0]) if years else None
        # Latest Games disciplines shown alongside the projection (upcoming side).
        if latest is not None:
            upcoming.extend(_discipline_boards_for_year(df, latest, historical=False))
        # Historical browser: the most recent N past Games as completed boards.
        for year in [int(y) for y in years[:HISTORICAL_GAMES_COUNT]]:
            completed.extend(_discipline_boards_for_year(df, year, historical=True))
    except Exception as exc:  # pragma: no cover
        logger.warning("Olympics discipline detail failed: %s", exc)

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
    p = build_live_upcoming_payload()
    print(json.dumps({
        "upcoming": len(p["upcoming"]),
        "completed": len(p["completed"]),
        "top5": [(x["playerName"], x["winProbability"]) for x in (p["upcoming"][0]["predictions"][:5] if p["upcoming"] else [])],
    }, indent=2, default=str))
