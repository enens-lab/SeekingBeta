"""Soccer data client.

Two free sources:
  * football-data.co.uk  -> historical results + goals (training data)
  * ESPN hidden soccer API -> upcoming/finished fixtures, names, logos
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .constants import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    ESPN_BASE_URL,
    FOOTBALL_DATA_BASE_URL,
    LEAGUE_CONFIGS,
    canonical_national_name,
    canonical_team_name,
    current_season_start,
    football_data_season_code,
)

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "sports" / "soccer"
# Local override for the football-data CSV cache (e.g. a scratch dir in tests or a
# dev run, so a run never writes into a shared data folder).
RAW_CACHE_DIR_ENV = "SOCCER_RAW_CACHE_DIR"

# football-data columns kept when keep_odds=True: de-vigged closing averages give the
# market baseline printed NEXT TO the track record (aggregates only; per-match odds
# are never published -- the football-data licence is unconfirmed).
FD_ODDS_COLUMNS = {
    "AvgCH": "close_avg_home", "AvgCD": "close_avg_draw", "AvgCA": "close_avg_away",
    "AvgC>2.5": "close_avg_over25", "AvgC<2.5": "close_avg_under25",
}


def _raw_cache_dir() -> Path:
    override = os.getenv(RAW_CACHE_DIR_ENV)
    return Path(override) if override else (DATA_ROOT / "raw")


def _season_cache_is_final(cache_path: Path, season_start: int) -> bool:
    """A past season's cached CSV is only trusted if it was written after that
    season ended (July 1 of start+1). A copy cached mid-season would otherwise be
    used forever with the second half of the season missing."""
    try:
        written = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return written >= datetime(int(season_start) + 1, 7, 1, tzinfo=timezone.utc)

INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


# --- football-data.co.uk (history) ------------------------------------------
def fetch_football_data_csv(
    fd_code: str, season_start: int, cache_dir: Path | None = None, cache: bool = True
) -> pd.DataFrame:
    """Download one league-season results CSV.

    Past seasons are cached to disk permanently; the current (in-progress)
    season should pass ``cache=False`` so freshly-played results are picked up.
    """
    code = football_data_season_code(season_start)
    url = f"{FOOTBALL_DATA_BASE_URL}/{code}/{fd_code}.csv"
    cache_dir = cache_dir or _raw_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fd_code}_{code}.csv"

    if cache and cache_path.exists() and _season_cache_is_final(cache_path, season_start):
        try:
            return pd.read_csv(cache_path, encoding="latin-1")
        except Exception:  # pragma: no cover - corrupt cache, re-fetch
            pass

    resp = _session().get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    if cache:
        cache_path.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(resp.content), encoding="latin-1")


def normalize_football_data(raw: pd.DataFrame, season_start: int | None = None, keep_odds: bool = False) -> pd.DataFrame:
    """One football-data.co.uk league-season CSV -> date, home, away (canonical),
    home_goals, away_goals, season_start [+ closing-odds columns when keep_odds].

    Only full-time goals are used as model inputs. Same-match statistics (shots,
    and from 2026-27 the post-match HxG/AxG) are never read."""
    cols = {str(c).strip().lower(): c for c in raw.columns}
    needed = ["hometeam", "awayteam", "fthg", "ftag"]
    if not all(n in cols for n in needed):
        return pd.DataFrame(columns=["date", "home", "away", "home_goals", "away_goals", "season_start"])
    frame = pd.DataFrame(
        {
            "home": raw[cols["hometeam"]].map(canonical_team_name),
            "away": raw[cols["awayteam"]].map(canonical_team_name),
            "home_goals": pd.to_numeric(raw[cols["fthg"]], errors="coerce"),
            "away_goals": pd.to_numeric(raw[cols["ftag"]], errors="coerce"),
            # football-data's own spelling, for display ("Nott'm Forest", not a
            # title-cased canonical key)
            "home_display": raw[cols["hometeam"]].astype(str).str.strip(),
            "away_display": raw[cols["awayteam"]].astype(str).str.strip(),
        }
    )
    if "date" in cols:
        text = raw[cols["date"]].astype(str)
        parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
        parsed = parsed.fillna(pd.to_datetime(text, format="%d/%m/%y", errors="coerce"))
        frame["date"] = parsed
    else:
        frame["date"] = pd.NaT
    frame["season_start"] = int(season_start) if season_start is not None else pd.NA
    if keep_odds:
        for src, dst in FD_ODDS_COLUMNS.items():
            frame[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else float("nan")
    frame = frame.dropna(subset=["home", "away", "home_goals", "away_goals"])
    return frame[(frame["home"] != "") & (frame["away"] != "")]


def load_history(league_key: str, seasons: int = 3, current_season_start: int | None = None,
                 keep_odds: bool = False) -> pd.DataFrame:
    """Concatenated, normalized match history for a league.

    Returns columns: date (datetime), home, away (canonical), home_goals,
    away_goals, season_start (int) [+ closing odds when keep_odds], sorted by date.
    The current season is derived from today's date (July rollover) unless given.
    """
    cfg = LEAGUE_CONFIGS.get(league_key)
    if not cfg or not cfg.get("fd_code"):
        raise ValueError(f"league {league_key!r} has no football-data history source")
    fd_code = str(cfg["fd_code"])
    base = current_season_start if current_season_start is not None else current_season_start_fn()

    frames: list[pd.DataFrame] = []
    for offset in range(seasons):
        start = base - offset
        try:
            # Always re-fetch the current (in-progress) season; cache older ones.
            raw = fetch_football_data_csv(fd_code, start, cache=(offset > 0))
        except Exception as exc:  # pragma: no cover - network/season gaps
            logger.warning("soccer history fetch failed for %s %s: %s", fd_code, start, exc)
            continue
        frame = normalize_football_data(raw, season_start=start, keep_odds=keep_odds)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "home", "away", "home_goals", "away_goals", "season_start"])

    out = pd.concat(frames, ignore_index=True)
    out["season_start"] = out["season_start"].astype(int)
    out = out.dropna(subset=["date"])
    return out.sort_values(["date", "home"], kind="mergesort").reset_index(drop=True)


current_season_start_fn = current_season_start


# --- international results (national teams, for the World Cup model) ---------
def fetch_international_results(since_year: int | None = None) -> pd.DataFrame:
    """Men's international match results (martj42, 1872->present).

    Returns columns: date, home, away (canonical national names), home_goals,
    away_goals, neutral (bool). Cached to disk; refreshed when stale is fine
    since the export endpoint caches its own payload.
    """
    cache_dir = _raw_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "international_results.csv"

    raw: pd.DataFrame | None = None
    try:
        resp = _session().get(INTERNATIONAL_RESULTS_URL, timeout=DEFAULT_TIMEOUT_SECONDS)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        raw = pd.read_csv(io.BytesIO(resp.content))
    except Exception as exc:  # pragma: no cover - fall back to cache
        logger.warning("international results fetch failed (%s); using cache", exc)
        if cache_path.exists():
            raw = pd.read_csv(cache_path)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "home", "away", "home_goals", "away_goals", "neutral", "tournament"])

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce"),
            "home": raw["home_team"].map(canonical_national_name),
            "away": raw["away_team"].map(canonical_national_name),
            "home_goals": pd.to_numeric(raw["home_score"], errors="coerce"),
            "away_goals": pd.to_numeric(raw["away_score"], errors="coerce"),
            "neutral": raw.get("neutral", False).astype(str).str.upper().eq("TRUE")
            if "neutral" in raw.columns
            else False,
            # Competition label ("FIFA World Cup", "Friendly", ...) for H2H/history.
            "tournament": raw["tournament"].astype(str) if "tournament" in raw.columns else "",
        }
    ).dropna(subset=["date", "home", "away", "home_goals", "away_goals"])

    if since_year is not None:
        frame = frame[frame["date"].dt.year >= since_year]
    return frame.sort_values("date").reset_index(drop=True)


# --- ESPN (live/upcoming fixtures) ------------------------------------------
def _espn_session() -> requests.Session:
    """ESPN's edge (Akamai) answers 403 to a browser-like User-Agent -- and to any
    custom token -- when the request comes from a datacenter IP (RunPod, Lightsail,
    EC2), but 200 to requests' own default UA. Verified 2026-09-19 from the web box:
    Chrome UA / "SeekingBeta/1.0" -> 403, "python-requests/2.32" / "curl" -> 200. So
    ESPN calls deliberately do NOT set the browser UA the rest of this client uses
    for football-data.co.uk."""
    return requests.Session()


def _month_keys(start: str, end: str) -> list[str]:
    """YYYYMM keys covering [start, end] (both YYYYMMDD)."""
    y, m = int(start[:4]), int(start[4:6])
    ey, em = int(end[:4]), int(end[4:6])
    keys = []
    while (y, m) <= (ey, em):
        keys.append(f"{y:04d}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return keys


def fetch_espn_scoreboard(espn_slug: str, dates: str | None = None) -> dict[str, Any]:
    """Raw ESPN scoreboard JSON. ``dates`` is YYYYMMDD, YYYYMM or YYYYMMDD-YYYYMMDD.

    ESPN stopped accepting the YYYYMMDD-YYYYMMDD range form (HTTP 400 from every
    network, 2026-09), which is what silently emptied the soccer upcoming boards.
    A range is now served by querying each covering month (YYYYMM, still accepted)
    and filtering events to the range client-side; the returned payload keeps the
    {"events": [...]} shape parse_espn_fixtures expects."""
    url = f"{ESPN_BASE_URL}/{espn_slug}/scoreboard"
    session = _espn_session()
    if dates and "-" in dates:
        start, end = dates.split("-", 1)
        events: dict[str, dict[str, Any]] = {}
        for month in _month_keys(start, end):
            resp = session.get(url, params={"dates": month}, timeout=DEFAULT_TIMEOUT_SECONDS)
            resp.raise_for_status()
            for event in resp.json().get("events", []) or []:
                day = str(event.get("date") or "")[:10].replace("-", "")
                if start <= day <= end:
                    events[str(event.get("id") or f"{day}-{len(events)}")] = event
        return {"events": list(events.values()), "range": [start, end], "months": _month_keys(start, end)}
    params = {"dates": dates} if dates else None
    resp = session.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def fetch_espn_team_index(espn_slug: str) -> dict[str, str]:
    """Map canonical national-team name -> ESPN team id, for the given league."""
    url = f"{ESPN_BASE_URL}/{espn_slug}/teams"
    resp = _espn_session().get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    index: dict[str, str] = {}
    try:
        teams = data["sports"][0]["leagues"][0]["teams"]
    except (KeyError, IndexError, TypeError):
        return index
    for entry in teams:
        team = entry.get("team") or {}
        name = team.get("displayName") or team.get("name")
        tid = team.get("id")
        if name and tid:
            index[canonical_national_name(name)] = str(tid)
    return index


def fetch_espn_roster(espn_slug: str, team_id: str) -> list[dict[str, Any]]:
    """Squad list for a team: [{name, position, age, number}]. Empty on failure."""
    url = f"{ESPN_BASE_URL}/{espn_slug}/teams/{team_id}/roster"
    try:
        resp = _espn_session().get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
        resp.raise_for_status()
        athletes = resp.json().get("athletes", []) or []
    except Exception as exc:  # pragma: no cover - network
        logger.warning("ESPN roster fetch failed for %s/%s: %s", espn_slug, team_id, exc)
        return []

    players: list[dict[str, Any]] = []
    for a in athletes:
        pos = a.get("position")
        pos_abbr = pos.get("abbreviation") if isinstance(pos, dict) else pos
        age = a.get("age")
        players.append({
            "name": a.get("displayName") or a.get("fullName") or "",
            "position": pos_abbr,
            "age": int(age) if isinstance(age, (int, float)) else None,
            "number": str(a.get("jersey")) if a.get("jersey") else None,
        })
    return [p for p in players if p["name"]]


def parse_espn_fixtures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ESPN scoreboard events into fixture dicts."""
    fixtures: list[dict[str, Any]] = []
    for event in payload.get("events", []) or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_c or not away_c:
            continue
        status = ((event.get("status") or {}).get("type") or {})
        date_raw = event.get("date") or ""  # e.g. 2026-05-30T14:00Z
        date_int = None
        if len(date_raw) >= 10:
            date_int = int(date_raw[0:10].replace("-", ""))
        venue = (comp.get("venue") or {}).get("fullName")
        start_iso = _espn_start_iso(date_raw)

        def _score(c: dict[str, Any]) -> int | None:
            try:
                return int(c.get("score"))
            except (TypeError, ValueError):
                return None

        fixtures.append(
            {
                "id": str(event.get("id")),
                "name": event.get("name") or event.get("shortName"),
                "date_int": date_int,
                "venue": venue,
                "completed": bool(status.get("completed")),
                "state": status.get("state"),  # pre | in | post
                "home_team": home_c.get("team") or {},
                "away_team": away_c.get("team") or {},
                "home_name": (home_c.get("team") or {}).get("displayName"),
                "away_name": (away_c.get("team") or {}).get("displayName"),
                "home_score": _score(home_c),
                "away_score": _score(away_c),
                # Kick-off in ISO-8601 UTC: the market pick log only grades a pick
                # published strictly before this (sports/market_log.py).
                "start_iso": start_iso,
                "neutral_site": comp.get("neutralSite"),
                "status_name": status.get("name"),
            }
        )
    return fixtures


def _espn_start_iso(date_raw: str) -> str | None:
    """'2026-10-10T11:30Z' -> '2026-10-10T11:30:00+00:00'. Date-only or unparseable
    values return None (a pick without a known kick-off is never graded)."""
    if not date_raw or "T" not in date_raw:
        return None
    try:
        ts = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()
