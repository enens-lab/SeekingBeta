"""WTA season results from ESPN's tennis scoreboard, written in Sackmann format.

Why: Jeff Sackmann's ``tennis_wta`` repository returns 404 (2026), tennismylife
only covers the ATP, and tennis-data.co.uk sits behind a Cloudflare challenge. So
without this the WTA ratings stop at the last results file on S3 (Nov 2025):
every WTA board would rate the 2026 field on ratings almost a year old, and no
WTA 2026 event could be backtested.

What is kept:

* completed women's singles main-draw matches (qualifying, doubles and the men's
  draw of combined events are dropped); walkovers keep a ``W/O`` score so the Elo
  skips them, retirements count as results (as in Sackmann's files);
* only tournaments that were tour-level in our history last season. ESPN's
  tournament ids are stable across years ("154-2026" is the Australian Open), so
  last season's ESPN events are matched to last season's tour-level events by
  their players (at least 60% of the larger field in common, start within 10
  days) and this season keeps the events whose tournament id matched. Sponsor
  renames and moved dates survive; WTA 125 / ITF events, which our history and
  model selection never included, are dropped. Last season's name, level and
  surface are carried over so a tournament keeps one identity;
* player names are resolved onto the ids already in our history (exact name,
  reversed order for surname-first spellings, or surname + initial when that is
  unique), so ratings carry over; anyone else gets an ``espn<id>`` id and is
  keyed on the name by ``elo.assign_player_keys``;
* the seed (``curatedRank``) becomes ``winner_seed``/``loser_seed``: ESPN gives
  no rankings, so the top seed stands in for the ranking favourite.

ESPN's site API is unofficial and Disney's terms restrict automated/commercial
use; the BFF and ``live_fields`` already read the same scoreboard. Set
``TENNIS_ESPN_RESULTS=0`` to turn this off (WTA ratings then stop at the last
results file, and ``ratingsAsOf`` on every board says so).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from sports.wta.elo import is_team_event, normalize_identifier, normalize_name
from sports.wta.live_fields import ESPN_TENNIS_SCOREBOARD, _date_key, _is_singles_for_tour

logger = logging.getLogger(__name__)

WINDOW_DAYS = 61
MIN_PRIOR_FIELD = 8
CROSSWALK_MIN_OVERLAP = 0.6
CROSSWALK_MAX_DAYS = 10
SOURCE_LABEL = "espn:scoreboard"
_ROUND_K_RE = re.compile(r"^round\s+(\d+)$", re.IGNORECASE)
_FIXED_ROUNDS = {"final": "F", "semifinal": "SF", "semifinals": "SF", "quarterfinal": "QF", "quarterfinals": "QF", "round robin": "RR"}
_ROUND_SORT = {"RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}
_SHORT_FORM = re.compile(r"^(?P<last>.+?)\s+(?P<init>(?:[A-Z]\.)+)$")

OUTPUT_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_name", "loser_id", "loser_seed", "loser_name", "score", "round",
    "winner_rank", "loser_rank", "tour", "data_source", "espn_event_id", "espn_event_name",
]


def espn_results_enabled() -> bool:
    return os.getenv("TENNIS_ESPN_RESULTS", "1").strip().lower() not in {"0", "false", "no", "off"}


# --------------------------------------------------------------------------- fetch


def fetch_season_events(
    tour: str,
    year: int,
    *,
    today: date | None = None,
    window_days: int = WINDOW_DAYS,
    timeout: float = 60.0,
    session: Any = None,
) -> list[dict[str, Any]] | None:
    """Every ESPN event of ``tour`` overlapping ``year`` (up to today), raw JSON.

    Starts mid-December of the year before: an event whose first round falls in
    late December belongs to the season of its final (Sackmann's convention).
    Returns None if any window fails: a season with a hole in it would silently
    drop results, so the caller keeps its previous file instead.
    """
    today = today or date.today()
    first = date(year - 1, 12, 15)
    last = min(date(year, 12, 31), today + timedelta(days=1))
    if last < first:
        return []
    getter = session or requests
    events: dict[str, dict[str, Any]] = {}
    cursor = first
    while cursor <= last:
        window_end = min(cursor + timedelta(days=window_days - 1), last)
        try:
            response = getter.get(
                ESPN_TENNIS_SCOREBOARD.format(tour=tour.lower()),
                params={"dates": f"{cursor:%Y%m%d}-{window_end:%Y%m%d}", "limit": 500},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # network / JSON
            logger.warning("ESPN %s results window %s-%s unavailable: %s", tour, cursor, window_end, exc)
            return None
        for event in payload.get("events") or []:
            event_id = str(event.get("id") or "")
            if event_id and event_id not in events:
                events[event_id] = event
        cursor = window_end + timedelta(days=1)
    return list(events.values())


# --------------------------------------------------------------------------- parse


def _clean_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tournament_key(event_id: Any) -> str:
    """ESPN event id "154-2026" -> tournament id "154" (stable across seasons)."""
    return str(event_id or "").split("-")[0]


def round_codes(labels: Iterable[str]) -> dict[str, str]:
    """ESPN main-draw round labels -> Sackmann codes ("Round 1" of a 7-round draw -> "R128")."""
    labels = [str(label or "").strip() for label in labels]
    numbered = [int(m.group(1)) for m in (_ROUND_K_RE.match(label) for label in labels) if m]
    deepest = max(numbered) if numbered else 0
    codes: dict[str, str] = {}
    for label in labels:
        fixed = _FIXED_ROUNDS.get(label.lower())
        if fixed:
            codes[label] = fixed
            continue
        match = _ROUND_K_RE.match(label)
        if match:
            codes[label] = f"R{2 ** (deepest - int(match.group(1)) + 4)}"
    return codes


def _score(competitors: list[dict[str, Any]], winner_index: int, status_name: str) -> str:
    if "WALKOVER" in status_name:
        return "W/O"
    winner = competitors[winner_index].get("linescores") or []
    loser = competitors[1 - winner_index].get("linescores") or []
    sets = []
    for won, lost in zip(winner, loser):
        try:
            text = f"{int(float(won.get('value')))}-{int(float(lost.get('value')))}"
        except (TypeError, ValueError):
            continue
        tiebreak = lost.get("tiebreak") if won.get("winner") else won.get("tiebreak")
        if tiebreak is not None:
            text += f"({tiebreak})"
        sets.append(text)
    score = " ".join(sets)
    if "RETIRED" in status_name:
        score = f"{score} RET".strip()
    return score


def _seed(competitor: dict[str, Any]) -> float:
    try:
        value = float((competitor.get("curatedRank") or {}).get("current"))
    except (TypeError, ValueError):
        return np.nan
    return value if 0 < value < 99 else np.nan


def parse_event_matches(event: dict[str, Any], tour: str) -> list[dict[str, Any]]:
    """Completed main-draw singles matches of one ESPN event for ``tour``."""
    out: list[dict[str, Any]] = []
    for grouping in event.get("groupings") or []:
        if not _is_singles_for_tour(grouping, tour):
            continue
        main = [
            comp for comp in grouping.get("competitions") or []
            if not str((comp.get("round") or {}).get("displayName") or "").lower().startswith("qualif")
        ]
        codes = round_codes({str((comp.get("round") or {}).get("displayName") or "") for comp in main})
        for comp in main:
            label = str((comp.get("round") or {}).get("displayName") or "").strip()
            code = codes.get(label)
            status = ((comp.get("status") or {}).get("type") or {})
            competitors = comp.get("competitors") or []
            if code is None or not status.get("completed") or len(competitors) != 2:
                continue
            winners = [index for index, competitor in enumerate(competitors) if competitor.get("winner")]
            if len(winners) != 1:
                continue
            w = winners[0]
            athletes = [competitor.get("athlete") or {} for competitor in competitors]
            names = [_clean_name(athlete.get("displayName") or athlete.get("fullName")) for athlete in athletes]
            if not all(names):
                continue
            out.append(
                {
                    "espn_event_id": str(event.get("id") or ""),
                    "espn_event_name": _clean_name(event.get("name")),
                    "match_date": _date_key(comp.get("date") or comp.get("startDate")) or _date_key(event.get("date")),
                    "round": code,
                    "competition_id": str(comp.get("id") or ""),
                    "winner_espn_id": str(competitors[w].get("id") or ""),
                    "winner_name": names[w],
                    "winner_seed": _seed(competitors[w]),
                    "loser_espn_id": str(competitors[1 - w].get("id") or ""),
                    "loser_name": names[1 - w],
                    "loser_seed": _seed(competitors[1 - w]),
                    "score": _score(competitors, w, str(status.get("name") or "").upper()),
                }
            )
    return out


def event_profiles(events: list[dict[str, Any]], tour: str, season: int) -> list[dict[str, Any]]:
    """Per ESPN event of ``season`` (= year of its last main-draw match): its matches,
    first/last match date and the surnames in its field."""
    profiles = []
    for event in events:
        matches = [m for m in parse_event_matches(event, tour) if m["match_date"]]
        if not matches:
            continue
        last = max(m["match_date"] for m in matches)
        if last // 10000 != season:
            continue
        players = {m["winner_name"] for m in matches} | {m["loser_name"] for m in matches}
        profiles.append(
            {
                "espn_id": str(event.get("id") or ""),
                "tournament": tournament_key(event.get("id")),
                "name": _clean_name(event.get("name")),
                "start": min(m["match_date"] for m in matches),
                "end": last,
                "surnames": {surname(name) for name in players} - {""},
                "matches": matches,
            }
        )
    return profiles


# --------------------------------------------------------------------------- identity


def surname(name: Any) -> str:
    """Normalised last surname token: "Haddad Maia B." and "Beatriz Haddad Maia" -> "maia"."""
    text = _clean_name(name)
    short = _SHORT_FORM.match(text)
    if short:
        text = short.group("last")
    tokens = text.split()
    return normalize_name(tokens[-1]) if tokens else ""


class PlayerResolver:
    """ESPN display name -> (player_id, player_name) already used in our history.

    In order, each step only when it identifies exactly one player:

    1. the exact normalised name, or the name with the surname moved (ESPN writes
       some names surname-first);
    2. surname + first initial for players we only know by a tennis-data short
       name ("Rakotomanga Rajaonah T."), trying every split of the ESPN name so
       multi-word surnames match;
    3. same last name and first names sharing their first three letters
       ("Catherine" / "Caty McNally"), among players active in the last two
       seasons of history.

    Ids are the legacy numeric ids ``elo.assign_player_keys`` keys on. Anyone not
    resolved keeps an ``espn<id>`` id and starts a new rating, which is the
    honest outcome for a genuine newcomer.
    """

    ACTIVE_YEARS = 2

    def __init__(self, prior: pd.DataFrame, tour: str):
        self.exact: dict[str, tuple[str, str]] = {}
        initials: dict[tuple[str, str], set[str]] = {}
        by_last: dict[str, dict[str, tuple[str, str]]] = {}
        last_seen: dict[str, int] = {}
        full_ids: set[str] = set()
        frame = prior[prior["tour"].astype(str).str.upper() == tour.upper()]
        dates = pd.to_numeric(frame["tourney_date"], errors="coerce").fillna(0).astype(int)
        for side in ("winner", "loser"):
            ids = frame[f"{side}_id"].map(normalize_identifier)
            for pid, name, day in zip(ids, frame[f"{side}_name"], dates):
                if not pid or not str(pid).isdigit():
                    continue
                last_seen[pid] = max(last_seen.get(pid, 0), int(day))
                text = _clean_name(name)
                short = _SHORT_FORM.match(text)
                if short:
                    initials.setdefault((normalize_name(short.group("last")), short.group("init")[0].lower()), set()).add(pid)
                    continue
                normalized = normalize_name(text)
                tokens = text.split()
                if normalized and len(tokens) >= 2:
                    self.exact[normalized] = (pid, text)  # later rows win: latest spelling
                    by_last.setdefault(normalize_name(tokens[-1]), {})[pid] = (normalize_name(tokens[0]), text)
                    full_ids.add(pid)
                elif normalized:
                    self.exact[normalized] = (pid, text)
                    full_ids.add(pid)
        self.initials: dict[tuple[str, str], str] = {
            parts: next(iter(ids)) for parts, ids in initials.items() if len(ids) == 1 and not ids & full_ids
        }
        horizon = (max(last_seen.values(), default=0) // 10000 - self.ACTIVE_YEARS + 1) * 10000
        self.by_last = {
            last: {pid: entry for pid, entry in players.items() if last_seen.get(pid, 0) >= horizon}
            for last, players in by_last.items()
        }

    def resolve(self, name: str) -> tuple[str, str] | None:
        """(id, name to write). Matches on our own full names keep our spelling so
        the key and display name never change; a short-name-only player gets
        ESPN's full name, which gives the boards a proper name for her."""
        clean = _clean_name(name)
        tokens = clean.split()
        candidates = [clean]
        if len(tokens) >= 2:
            candidates.append(" ".join(tokens[1:] + tokens[:1]))
            candidates.append(" ".join(tokens[-1:] + tokens[:-1]))
        for candidate in candidates:
            found = self.exact.get(normalize_name(candidate))
            if found:
                return found
        if len(tokens) < 2:
            return None
        initial = normalize_name(tokens[0])[:1]
        for split in range(1, len(tokens)):
            pid = self.initials.get((normalize_name(" ".join(tokens[split:])), initial))
            if pid:
                return pid, clean
        first = normalize_name(tokens[0])
        same_last = self.by_last.get(normalize_name(tokens[-1]), {})
        close = [entry for pid, entry in same_last.items() if len(first) >= 3 and entry[0][:3] == first[:3]]
        if len(close) == 1:
            pid = next(pid for pid, entry in same_last.items() if entry is close[0])
            return pid, close[0][1]
        return None


# --------------------------------------------------------------------------- tour level


def prior_editions(prior: pd.DataFrame, tour: str, season: int) -> pd.DataFrame:
    """Individual tour-level events of ``season`` in our history, with their field's surnames."""
    columns = ["tourney_id", "tourney_name", "tourney_level", "surface", "draw_size", "start", "players", "surnames"]
    frame = prior[prior["tour"].astype(str).str.upper() == tour.upper()].copy()
    for column in ("tourney_level", "surface", "draw_size"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["tourney_date"] = pd.to_numeric(frame["tourney_date"], errors="coerce")
    frame = frame.dropna(subset=["tourney_date"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    # Season of an event = year of its last match (late-December starts belong to the next season).
    frame = frame[frame.groupby("tourney_id")["tourney_date"].transform("max").astype(int) // 10000 == season]
    frame = frame[[not is_team_event(level, name) for level, name in zip(frame["tourney_level"], frame["tourney_name"])]]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for tourney_id, group in frame.groupby("tourney_id", sort=False):
        names = set(group["winner_name"].astype(str)) | set(group["loser_name"].astype(str))
        rows.append(
            {
                "tourney_id": tourney_id,
                "tourney_name": str(group["tourney_name"].iloc[0]),
                "tourney_level": group["tourney_level"].iloc[0],
                "surface": group["surface"].iloc[0],
                "draw_size": group["draw_size"].iloc[0],
                "start": int(group["tourney_date"].min()),
                "players": len(names),
                "surnames": {surname(name) for name in names} - {""},
            }
        )
    editions = pd.DataFrame(rows, columns=columns)
    return editions[editions["players"] >= MIN_PRIOR_FIELD].sort_values("players", ascending=False).reset_index(drop=True)


def _days_between(a: int, b: int) -> int:
    to_date = lambda key: date(key // 10000, (key // 100) % 100, key % 100)  # noqa: E731
    return abs((to_date(int(a)) - to_date(int(b))).days)


def tour_level_crosswalk(prior_profiles: list[dict[str, Any]], editions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """ESPN tournament id -> our tour-level edition (name, level, surface) of the
    same event last season, matched on the players both fields share."""
    crosswalk: dict[str, dict[str, Any]] = {}
    taken: set[str] = set()
    for edition in editions.itertuples(index=False):
        best, best_score = None, 0.0
        for profile in prior_profiles:
            if profile["espn_id"] in taken or not profile["surnames"] or not edition.surnames:
                continue
            if _days_between(profile["start"], edition.start) > CROSSWALK_MAX_DAYS:
                continue
            # Shared players over the LARGER field: a 28-player warm-up event is
            # almost a subset of the Wimbledon field the week after, so dividing by
            # the smaller field would pair them.
            score = len(profile["surnames"] & edition.surnames) / max(len(profile["surnames"]), len(edition.surnames))
            if score > best_score:
                best, best_score = profile, score
        if best is None or best_score < CROSSWALK_MIN_OVERLAP:
            continue
        taken.add(best["espn_id"])
        crosswalk.setdefault(
            best["tournament"],
            {
                "tourney_name": edition.tourney_name,
                "tourney_level": edition.tourney_level,
                "surface": edition.surface,
                "draw_size": edition.draw_size,
                "matched_espn_event": best["name"],
                "overlap": round(best_score, 3),
            },
        )
    return crosswalk


def crosswalk_from_espn_rows(prior: pd.DataFrame, tour: str, season: int) -> dict[str, dict[str, Any]]:
    """The same crosswalk read straight from an earlier ESPN-sourced season (its
    rows carry ``espn_event_id``), so no second season has to be fetched."""
    if "espn_event_id" not in prior.columns:
        return {}
    ids = prior["espn_event_id"].astype(str).str.strip()
    frame = prior[(prior["tour"].astype(str).str.upper() == tour.upper()) & ~ids.str.lower().isin({"", "nan", "none", "<na>"})].copy()
    frame["tourney_date"] = pd.to_numeric(frame["tourney_date"], errors="coerce")
    frame = frame[frame["tourney_date"] // 10000 == season]
    crosswalk: dict[str, dict[str, Any]] = {}
    for event_id, group in frame.groupby("espn_event_id", sort=False):
        key = tournament_key(event_id)
        if key:
            crosswalk.setdefault(
                key,
                {
                    "tourney_name": str(group["tourney_name"].iloc[0]),
                    "tourney_level": group["tourney_level"].iloc[0] if "tourney_level" in group else np.nan,
                    "surface": group["surface"].iloc[0] if "surface" in group else np.nan,
                    "draw_size": group["draw_size"].iloc[0] if "draw_size" in group else np.nan,
                    "matched_espn_event": str(group["espn_event_name"].iloc[0]) if "espn_event_name" in group else "",
                    "overlap": 1.0,
                },
            )
    return crosswalk


# --------------------------------------------------------------------------- build


def build_season_frame(
    profiles: list[dict[str, Any]],
    crosswalk: dict[str, dict[str, Any]],
    prior: pd.DataFrame,
    *,
    tour: str,
    year: int,
) -> pd.DataFrame:
    """Sackmann-format rows for ``year`` from parsed ESPN events (see module docstring)."""
    resolver = PlayerResolver(prior, tour)
    rows: list[dict[str, Any]] = []
    kept, dropped = [], []
    for profile in profiles:
        edition = crosswalk.get(profile["tournament"])
        if edition is None:
            dropped.append(profile["name"])
            continue
        kept.append(profile["name"])
        matches = sorted(profile["matches"], key=lambda m: (_ROUND_SORT.get(m["round"], 0), m["match_date"], m["competition_id"]))
        for number, match in enumerate(matches, start=1):
            row = {
                "tourney_id": f"{year}-ESPN-{profile['tournament']}",
                "tourney_name": edition["tourney_name"],
                "surface": edition["surface"],
                "draw_size": edition["draw_size"],
                "tourney_level": edition["tourney_level"],
                "tourney_date": int(profile["start"]),
                "match_num": number,
                "score": match["score"],
                "round": match["round"],
                "winner_rank": np.nan,
                "loser_rank": np.nan,
                "tour": tour.upper(),
                "data_source": SOURCE_LABEL,
                "espn_event_id": profile["espn_id"],
                "espn_event_name": profile["name"],
            }
            for side in ("winner", "loser"):
                resolved = resolver.resolve(match[f"{side}_name"])
                if resolved:
                    row[f"{side}_id"], row[f"{side}_name"] = resolved
                else:
                    row[f"{side}_id"] = f"espn{match[f'{side}_espn_id']}"
                    row[f"{side}_name"] = match[f"{side}_name"]
                row[f"{side}_seed"] = match[f"{side}_seed"]
            rows.append(row)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    logger.info(
        "ESPN %s %s: %d matches from %d tour-level events; %d other events dropped (%s)",
        tour, year, len(frame), len(kept), len(dropped), "; ".join(sorted(dropped)[:15]),
    )
    return frame


def season_matches(
    tour: str,
    year: int,
    prior: pd.DataFrame,
    *,
    today: date | None = None,
    fetch=fetch_season_events,
) -> pd.DataFrame | None:
    """Fetch + build ``year``. None when disabled, ESPN is unreachable or last
    season cannot be cross-walked (the caller then keeps whatever file it has)."""
    if not espn_results_enabled():
        return None
    crosswalk = crosswalk_from_espn_rows(prior, tour, year - 1)
    if not crosswalk:
        prior_events = fetch(tour, year - 1, today=today)
        if prior_events is None:
            return None
        crosswalk = tour_level_crosswalk(event_profiles(prior_events, tour, year - 1), prior_editions(prior, tour, year - 1))
    if not crosswalk:
        logger.warning("ESPN %s %s: no tour-level crosswalk from %s; not writing results.", tour, year, year - 1)
        return None
    events = fetch(tour, year, today=today)
    if events is None:
        return None
    return build_season_frame(event_profiles(events, tour, year), crosswalk, prior, tour=tour, year=year)
