"""Static Basketball constants used by NBA/WNBA ingestion and modeling."""

from __future__ import annotations

from dataclasses import dataclass
import re

DEFAULT_TIMEOUT_SECONDS = 30
# Must be a CLEAN browser UA — the NBA/WNBA CDN (Akamai) bot-detection 403s a UA with
# a non-standard token appended (the old " SeekingBetaAI/1.0" suffix tripped WNBA).
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class LeagueConfig:
    key: str
    display_name: str
    domain: str
    league_id: str
    current_season: str
    regular_season_game_prefix: str
    archive_max_game_number: int


LEAGUE_CONFIGS: dict[str, LeagueConfig] = {
    "nba": LeagueConfig(
        key="nba",
        display_name="Basketball (NBA)",
        domain="https://cdn.nba.com",
        league_id="00",
        current_season="2025-26",
        regular_season_game_prefix="002",
        archive_max_game_number=1500,
    ),
    "wnba": LeagueConfig(
        key="wnba",
        display_name="Basketball (WNBA)",
        domain="https://cdn.wnba.com",
        league_id="10",
        current_season="2026",
        regular_season_game_prefix="102",
        archive_max_game_number=500,
    ),
}

SCHEDULE_PATH = "/static/json/staticData/scheduleLeagueV2_1.json"
TODAY_SCOREBOARD_TEMPLATE = "/static/json/liveData/scoreboard/todaysScoreboard_{league_id}.json"
BOXSCORE_TEMPLATE = "/static/json/liveData/boxscore/boxscore_{game_id}.json"

DEFAULT_LEAGUES = ("nba", "wnba")

_NBA_SEASON_RE = re.compile(r"^(?P<start>\d{4})(?:-(?P<end>\d{2,4}))?$")
_WNBA_SEASON_RE = re.compile(r"^(?P<season>\d{4})$")


def season_start_year_from_label(league: str, season_label: str | int) -> int:
    text = str(season_label).strip()
    if league == "nba":
        match = _NBA_SEASON_RE.match(text)
        if not match:
            raise ValueError(f"Invalid NBA season label: {season_label}")
        return int(match.group("start"))
    if league == "wnba":
        match = _WNBA_SEASON_RE.match(text)
        if not match:
            raise ValueError(f"Invalid WNBA season label: {season_label}")
        return int(match.group("season"))
    raise ValueError(f"Unsupported basketball league: {league}")


def normalize_season_label(league: str, season_label: str | int) -> str:
    start_year = season_start_year_from_label(league, season_label)
    if league == "nba":
        return f"{start_year}-{str(start_year + 1)[-2:]}"
    if league == "wnba":
        return str(start_year)
    raise ValueError(f"Unsupported basketball league: {league}")


def archive_game_id_prefix(league: str, season_label: str | int) -> str:
    config = LEAGUE_CONFIGS[league]
    start_year = season_start_year_from_label(league, season_label)
    return f"{config.regular_season_game_prefix}{str(start_year)[-2:]}"


def season_labels_for_year_range(league: str, start_year: int, end_year: int) -> list[str]:
    if end_year < start_year:
        raise ValueError("season_end_year must be greater than or equal to season_start_year")
    return [normalize_season_label(league, year) for year in range(start_year, end_year + 1)]
