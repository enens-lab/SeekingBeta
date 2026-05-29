"""Soccer data client.

Two free sources:
  * football-data.co.uk  -> historical results + goals (training data)
  * ESPN hidden soccer API -> upcoming/finished fixtures, names, logos
"""

from __future__ import annotations

import io
import logging
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
    canonical_team_name,
    football_data_season_code,
)

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "sports" / "soccer"


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
    cache_dir = cache_dir or (DATA_ROOT / "raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fd_code}_{code}.csv"

    if cache and cache_path.exists():
        try:
            return pd.read_csv(cache_path, encoding="latin-1")
        except Exception:  # pragma: no cover - corrupt cache, re-fetch
            pass

    resp = _session().get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    if cache:
        cache_path.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(resp.content), encoding="latin-1")


def load_history(league_key: str, seasons: int = 3, current_season_start: int | None = None) -> pd.DataFrame:
    """Concatenated, normalized match history for a league.

    Returns columns: date (datetime), home, away (canonical), home_goals,
    away_goals. Newest season first in the source, sorted ascending by date.
    """
    from .constants import CURRENT_SEASON_START

    cfg = LEAGUE_CONFIGS.get(league_key)
    if not cfg or not cfg.get("fd_code"):
        raise ValueError(f"league {league_key!r} has no football-data history source")
    fd_code = str(cfg["fd_code"])
    base = current_season_start if current_season_start is not None else CURRENT_SEASON_START

    frames: list[pd.DataFrame] = []
    for offset in range(seasons):
        start = base - offset
        try:
            # Always re-fetch the current (in-progress) season; cache older ones.
            raw = fetch_football_data_csv(fd_code, start, cache=(offset > 0))
        except Exception as exc:  # pragma: no cover - network/season gaps
            logger.warning("soccer history fetch failed for %s %s: %s", fd_code, start, exc)
            continue
        cols = {c.lower(): c for c in raw.columns}
        needed = ["hometeam", "awayteam", "fthg", "ftag"]
        if not all(n in cols for n in needed):
            continue
        frame = pd.DataFrame(
            {
                "home": raw[cols["hometeam"]].map(canonical_team_name),
                "away": raw[cols["awayteam"]].map(canonical_team_name),
                "home_goals": pd.to_numeric(raw[cols["fthg"]], errors="coerce"),
                "away_goals": pd.to_numeric(raw[cols["ftag"]], errors="coerce"),
            }
        )
        if "date" in cols:
            frame["date"] = pd.to_datetime(raw[cols["date"]], dayfirst=True, errors="coerce")
        else:
            frame["date"] = pd.NaT
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "home", "away", "home_goals", "away_goals"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["home", "away", "home_goals", "away_goals"])
    out = out[(out["home"] != "") & (out["away"] != "")]
    return out.sort_values("date").reset_index(drop=True)


# --- ESPN (live/upcoming fixtures) ------------------------------------------
def fetch_espn_scoreboard(espn_slug: str, dates: str | None = None) -> dict[str, Any]:
    """Raw ESPN scoreboard JSON. ``dates`` is YYYYMMDD or YYYYMMDD-YYYYMMDD."""
    url = f"{ESPN_BASE_URL}/{espn_slug}/scoreboard"
    params = {"dates": dates} if dates else None
    resp = _session().get(url, params=params, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


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
            }
        )
    return fixtures
