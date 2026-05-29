"""Soccer (association football) frontend data export.

Phase 0 scaffold: this module exposes the same ``build_live_upcoming_payload``
contract that the divination soccer endpoint (``/api/sports/soccer/boards``)
and the prophecy BFF expect, but returns an empty-but-valid payload until the
Phase 1 ingestion + Dixon-Coles model lands.

Soccer is modelled as ONE sport key whose ``tour`` field carries the
competition (e.g. "Premier League", "La Liga", "Serie A", "Bundesliga",
"Ligue 1", "UEFA Champions League", "FIFA World Cup"). This mirrors how golf
uses ``tour`` for PGA/LPGA and tennis for ATP/WTA, so every competition rides
on a single set of API / web / iOS plumbing.

DRAW / 1X2 CONTRACT
-------------------
Unlike the binary (home-win) team sports already in the platform, soccer has
three outcomes. Each board therefore carries an explicit 1X2 distribution:

    {
        "id": "epl-2026-08-15-ARS-CHE",
        "name": "Arsenal vs Chelsea",
        "tour": "Premier League",
        "course": "",                # unused for soccer; kept for schema parity
        "venue": "Emirates Stadium",
        "scheduledDate": 20260815,
        "latestDate": 20260815,
        "awayTeam": "Chelsea",
        "homeTeam": "Arsenal",
        "predictedWinner": "Arsenal",   # argmax label, or "Draw"
        "homeWinProbability": 0.52,      # NEW (1X2)
        "drawProbability": 0.26,         # NEW (1X2)
        "awayWinProbability": 0.22,      # NEW (1X2)
        "awayTeamDetails": {...},        # logo/colors via branding
        "homeTeamDetails": {...},
        "predictions": [                 # also emitted for backward-compat clients
            {"rank": 1, "playerName": "Arsenal", "winProbability": 52.0, "side": "home"},
            {"rank": 2, "playerName": "Draw",    "winProbability": 26.0, "side": "draw"},
            {"rank": 3, "playerName": "Chelsea", "winProbability": 22.0, "side": "away"},
        ],
    }

The ``homeWinProbability`` / ``drawProbability`` / ``awayWinProbability`` fields
are Optional in the prophecy/iOS/web contracts, so existing binary sports that
omit them are unaffected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LIVE_SOURCE = "divination_live_soccer_feed"


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
    """Assemble a single draw-aware soccer board dict.

    Probabilities are accepted as 0..1 floats; the ``predictions`` mirror uses
    0..100 percentages to match the existing ranked-field convention.
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
    """Return the live soccer board payload.

    Phase 0: returns an empty-but-valid payload so the soccer endpoint and the
    soccer tab work end-to-end before the model exists. Phase 1 will replace the
    body with real fixture loading + Dixon-Coles 1X2 predictions, using
    ``build_soccer_board(...)`` for each match and grouping by ``tour``.
    """

    # TODO(phase-1): load fixtures (ESPN hidden API / football-data.org), run the
    # Dixon-Coles model per competition, and emit boards via build_soccer_board().
    return _empty_payload(selected_date)


if __name__ == "__main__":
    import json

    print(json.dumps(build_live_upcoming_payload(), indent=2))
