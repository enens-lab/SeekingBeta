"""Discipline + medalist detail for the Summer Olympics, from the 120-year
athlete-event dataset (Sport, Event, Year, Name, NOC, Medal).

Builds: per-sport discipline lists with medalists, for a given Games year, plus
a best-effort athlete head-to-head (athletes who medaled in the same event).
"""

from __future__ import annotations

import re

from typing import Any

import pandas as pd

from .constants import noc_display_name

_MEDAL_ORDER = {"Gold": 0, "Silver": 1, "Bronze": 2}


def latest_summer_year(df: pd.DataFrame) -> int:
    summer = df[df["Season"] == "Summer"]
    return int(summer["Year"].max())


def sports_for_year(df: pd.DataFrame, year: int) -> list[str]:
    s = df[(df["Season"] == "Summer") & (df["Year"] == year)]
    return sorted(s["Sport"].dropna().unique().tolist())


def disciplines_for_sport(df: pd.DataFrame, sport: str, year: int) -> list[dict[str, Any]]:
    """Each event under a sport for a Games year, with its medalists."""
    rows = df[
        (df["Season"] == "Summer")
        & (df["Year"] == year)
        & (df["Sport"] == sport)
        & (df["Medal"].notna())
    ]
    out: list[dict[str, Any]] = []
    for event, grp in rows.groupby("Event"):
        # one medal per (Event, Medal, NOC) — collapse team-event duplicate rows
        seen: set[tuple[str, str]] = set()
        medalists: list[dict[str, Any]] = []
        for r in grp.itertuples(index=False):
            key = (str(r.Medal), str(r.NOC))
            if key in seen:
                continue
            seen.add(key)
            medalists.append({
                "medal": str(r.Medal),
                "name": str(r.Name),
                "country": noc_display_name(str(r.NOC)),
            })
        medalists.sort(key=lambda m: _MEDAL_ORDER.get(m["medal"], 9))
        out.append({"sport": sport, "event": str(event), "year": int(year), "medalists": medalists})
    out.sort(key=lambda d: d["event"])
    return out


def _name_matches(series: pd.Series, query: str) -> pd.Series:
    """Match athletes by all whitespace-separated tokens of the query appearing
    in the stored (full legal) name, case-insensitive. The dataset stores names
    like 'Ryan Steven Lochte', so exact-equality fails; token-AND is robust."""
    tokens = [t for t in str(query).lower().split() if len(t) > 1]
    if not tokens:
        return pd.Series(False, index=series.index)
    low = series.fillna("").str.lower()
    mask = pd.Series(True, index=series.index)
    for t in tokens:
        mask &= low.str.contains(re.escape(t), na=False)
    return mask


def athlete_head_to_head(df: pd.DataFrame, name_a: str, name_b: str) -> dict[str, Any]:
    """Best-effort: events where BOTH athletes competed (same Sport+Event+Year),
    with each one's medal. Approximate — the dataset is per-event participation,
    not bracket matchups, so this surfaces shared events, not direct results."""
    summer = df[df["Season"] == "Summer"]
    a = summer[_name_matches(summer["Name"], name_a)]
    b = summer[_name_matches(summer["Name"], name_b)]
    a_ev = set(zip(a["Year"], a["Event"]))
    b_ev = set(zip(b["Year"], b["Event"]))
    shared = sorted(a_ev & b_ev, reverse=True)
    rows = []
    for year, event in shared:
        am = a[(a["Year"] == year) & (a["Event"] == event)]["Medal"].dropna()
        bm = b[(b["Year"] == year) & (b["Event"] == event)]["Medal"].dropna()
        rows.append({
            "year": int(year), "event": str(event),
            "aMedal": str(am.iloc[0]) if len(am) else None,
            "bMedal": str(bm.iloc[0]) if len(bm) else None,
        })
    return {"athleteA": name_a, "athleteB": name_b, "sharedEvents": rows,
            "note": "Shared Olympic events (per-event participation; not a direct bracket)."}
