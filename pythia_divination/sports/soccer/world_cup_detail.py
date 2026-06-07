"""Head-to-head, recent form, and past-World-Cup detail for national teams.

All derived from the martj42 international-results dataset (date, home, away,
goals, neutral, tournament). Builds the optional detail blocks the board
contract carries for FIFA World Cup match boards.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import canonical_national_name

MAX_H2H_MATCHES = 8
MAX_FORM_RESULTS = 6


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in str(name).split())


def head_to_head(results: pd.DataFrame, team_a: str, team_b: str) -> dict[str, Any]:
    """All-time meetings between two nations, newest first."""
    a = canonical_national_name(team_a)
    b = canonical_national_name(team_b)
    mask = ((results["home"] == a) & (results["away"] == b)) | (
        (results["home"] == b) & (results["away"] == a)
    )
    meetings = results[mask].sort_values("date", ascending=False)
    if meetings.empty:
        return {"summary": f"No prior meetings between {_title(a)} and {_title(b)}.",
                "homeWins": 0, "awayWins": 0, "draws": 0, "matches": []}

    a_wins = b_wins = draws = 0
    rows: list[dict[str, Any]] = []
    for r in meetings.itertuples(index=False):
        hg, ag = int(r.home_goals), int(r.away_goals)
        if hg == ag:
            draws += 1
        elif (r.home == a) == (hg > ag):  # team a won
            a_wins += 1
        else:
            b_wins += 1
        if len(rows) < MAX_H2H_MATCHES:
            rows.append({
                "date": r.date.strftime("%Y-%m-%d") if not pd.isna(r.date) else None,
                "homeTeam": _title(r.home),
                "awayTeam": _title(r.away),
                "homeScore": hg,
                "awayScore": ag,
                "competition": str(getattr(r, "tournament", "") or "") or None,
            })
    return {
        "summary": f"{_title(a)} {a_wins}-{draws}-{b_wins} {_title(b)} ({len(meetings)} meetings)",
        "homeWins": a_wins,
        "awayWins": b_wins,
        "draws": draws,
        "matches": rows,
    }


def team_history(results: pd.DataFrame, team: str) -> dict[str, Any]:
    """Recent form + past World Cup appearances for one nation."""
    t = canonical_national_name(team)
    played = results[(results["home"] == t) | (results["away"] == t)].sort_values("date", ascending=False)

    recent_results: list[dict[str, Any]] = []
    form_chars: list[str] = []
    for r in played.head(MAX_FORM_RESULTS).itertuples(index=False):
        is_home = r.home == t
        gf, ga = (int(r.home_goals), int(r.away_goals)) if is_home else (int(r.away_goals), int(r.home_goals))
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        form_chars.append(res)
        recent_results.append({
            "date": r.date.strftime("%Y-%m-%d") if not pd.isna(r.date) else None,
            "opponent": _title(r.away if is_home else r.home),
            "result": res,
            "score": f"{gf}-{ga}",
            "competition": str(getattr(r, "tournament", "") or "") or None,
        })

    # Past World Cup *finals* participation (exclude "...World Cup qualification").
    tour_l = played["tournament"].astype(str).str.lower()
    wc = played[tour_l.eq("fifa world cup")]
    years = sorted({int(d.year) for d in wc["date"] if not pd.isna(d)}, reverse=True)
    past = [f"{y}: appeared" for y in years[:6]]

    return {
        "team": _title(t),
        "recentForm": "".join(form_chars),
        "recentResults": recent_results,
        "pastTournament": past,
    }
