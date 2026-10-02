"""Current tournament dates and main-draw fields from the ESPN tennis scoreboard.

Used only to (a) date upcoming boards and (b) replace last year's field with the
real main draw once it is published. The pricing and probabilities never come
from ESPN. Licensing: ESPN's site API is unofficial and Disney's terms restrict
automated/commercial use; the BFF already reads the same endpoint for tennis
dates (pythia_prophecy/api/service.py::_fetch_espn_tennis_schedule). Set
``TENNIS_ESPN_FIELDS=0`` to turn this off; boards then fall back to the
previous edition's field and say so in ``fieldBasis``.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import date, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

ESPN_TENNIS_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
_GENDER_FOR_TOUR = {"ATP": "men", "WTA": "women"}
_PLACEHOLDER_NAMES = {"", "tbd", "bye", "qualifier", "lucky loser", "q", "ll", "to be determined"}
_STOPWORDS = {
    "open", "championships", "championship", "tennis", "masters", "international", "internationals",
    "cup", "classic", "tournament", "atp", "wta", "presented", "by", "the", "de", "of", "du", "la",
    "group", "and", "women", "womens", "men", "mens", "ladies", "trophy", "grand", "prix", "tour",
    "series", "finals", "final", "rolex", "bnp", "paribas", "fortis", "mutua", "city", "pr",
}


def espn_enabled() -> bool:
    return os.getenv("TENNIS_ESPN_FIELDS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _ascii(text: Any) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()


def name_tokens(*texts: Any) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in re.split(r"[^a-z]+", _ascii(text).lower()):
            if len(token) >= 3 and token not in _STOPWORDS:
                tokens.add(token)
    return tokens


def _date_key(value: Any) -> int | None:
    text = str(value or "")[:10].replace("-", "")
    return int(text) if len(text) == 8 and text.isdigit() else None


def _is_singles_for_tour(grouping: dict[str, Any], tour: str) -> bool:
    label = str((grouping.get("grouping") or {}).get("displayName") or "").lower()
    if "singles" not in label:
        return False
    gender = _GENDER_FOR_TOUR.get(tour.upper())
    if gender == "men":
        return label.startswith("men")
    if gender == "women":
        return label.startswith("women")
    return True


def parse_scoreboard_events(payload: dict[str, Any], tour: str) -> list[dict[str, Any]]:
    """Events for one tour: name, venue, dates and the named main-draw entrants.

    ``entrants`` lists every named player in a main-draw (non-qualifying) match;
    ``open_slots`` counts first-round places still TBD (qualifiers not yet known).
    ``started`` is True once any main-draw match has a result.
    """
    events: list[dict[str, Any]] = []
    for event in payload.get("events") or []:
        name = str(event.get("name") or "").strip()
        start = _date_key(event.get("date"))
        end = _date_key(event.get("endDate")) or start
        if not name or not start:
            continue
        venue = str((event.get("venue") or {}).get("displayName") or "").strip()
        entrants: list[str] = []
        seen: set[str] = set()
        first_round_slots = 0
        open_slots = 0
        started = False
        first_round_label = None
        for grouping in event.get("groupings") or []:
            if not _is_singles_for_tour(grouping, tour):
                continue
            competitions = grouping.get("competitions") or []
            main = [
                comp for comp in competitions
                if not str((comp.get("round") or {}).get("displayName") or "").lower().startswith("qualif")
            ]
            if main:
                first_round_label = str((main[0].get("round") or {}).get("displayName") or "")
            for comp in main:
                round_label = str((comp.get("round") or {}).get("displayName") or "")
                state = str(((comp.get("status") or {}).get("type") or {}).get("state") or "")
                if state in {"in", "post"}:
                    started = True
                for competitor in comp.get("competitors") or []:
                    athlete = competitor.get("athlete") or {}
                    display = re.sub(r"\s+", " ", str(athlete.get("displayName") or "")).strip()
                    if round_label == first_round_label:
                        first_round_slots += 1
                        if display.lower() in _PLACEHOLDER_NAMES:
                            open_slots += 1
                    if display.lower() in _PLACEHOLDER_NAMES:
                        continue
                    key = display.lower()
                    if key not in seen:
                        seen.add(key)
                        entrants.append(display)
        events.append(
            {
                "name": name,
                "tour": tour.upper(),
                "venue": venue,
                "startKey": start,
                "endKey": end,
                "entrants": entrants,
                "firstRoundSlots": first_round_slots,
                "openSlots": open_slots,
                "started": started,
                "espnId": str(event.get("id") or ""),
            }
        )
    return events


def fetch_espn_events(tours: tuple[str, ...] = ("ATP", "WTA"), *, today: date | None = None,
                      days_back: int = 10, days_ahead: int = 75, timeout: float = 20.0) -> list[dict[str, Any]]:
    """All ESPN tennis events overlapping [today - days_back, today + days_ahead]. Never raises."""
    if not espn_enabled():
        return []
    today = today or date.today()
    window = f"{today - timedelta(days=days_back):%Y%m%d}-{today + timedelta(days=days_ahead):%Y%m%d}"
    events: list[dict[str, Any]] = []
    for tour in tours:
        try:
            response = requests.get(
                ESPN_TENNIS_SCOREBOARD.format(tour=tour.lower()),
                params={"dates": window, "limit": 300},
                timeout=timeout,
            )
            response.raise_for_status()
            events.extend(parse_scoreboard_events(response.json(), tour))
        except Exception as exc:  # pragma: no cover - network
            logger.warning("ESPN %s tennis scoreboard unavailable: %s", tour, exc)
    return events


def match_espn_event(
    tour: str,
    tournament_name: str,
    projected_start: int,
    espn_events: list[dict[str, Any]],
    *,
    max_days: int = 14,
) -> dict[str, Any] | None:
    """The ESPN event for one of our tournaments: same tour, start within
    ``max_days`` of the projected date, sharing a name or venue-city token."""
    ours = name_tokens(tournament_name)
    if not ours:
        return None
    try:
        projected = date(projected_start // 10000, (projected_start // 100) % 100, projected_start % 100)
    except ValueError:
        return None
    best: tuple[int, int, dict[str, Any]] | None = None
    for event in espn_events:
        if event.get("tour") != tour.upper():
            continue
        start = int(event["startKey"])
        try:
            start_date = date(start // 10000, (start // 100) % 100, start % 100)
        except ValueError:
            continue
        distance = abs((start_date - projected).days)
        if distance > max_days:
            continue
        city = str(event.get("venue") or "").split(",")[0]
        overlap = 2 * len(ours & name_tokens(event.get("name"))) + len(ours & name_tokens(city))
        if overlap == 0:
            continue
        candidate = (overlap, -distance, event)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best else None
