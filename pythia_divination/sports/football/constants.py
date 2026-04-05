"""Static Football constants for nflverse-backed ingestion and modeling."""

from __future__ import annotations

DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)

SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
TEAM_WEEK_STATS_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_team/"
    "stats_team_week_{season}.csv"
)
PLAYER_WEEK_STATS_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
    "stats_player_week_{season}.csv"
)
WEEKLY_ROSTERS_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/"
    "roster_weekly_{season}.csv"
)
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"

LEAGUE_KEY = "nfl"
DISPLAY_NAME = "Football"
CURRENT_SEASON = 2025
DEFAULT_SEASON_START = 2020
DEFAULT_SEASON_END = 2025
REGULAR_SEASON_TYPE = "REG"
PLAYOFF_SEASON_TYPE = "POST"
SUPPORTED_SEASON_TYPES = {REGULAR_SEASON_TYPE, PLAYOFF_SEASON_TYPE}
REGULAR_SEASON_WEEKS = tuple(range(1, 19))


def season_years_for_range(start_year: int, end_year: int) -> list[int]:
    if end_year < start_year:
        raise ValueError("season_end_year must be greater than or equal to season_start_year")
    return list(range(int(start_year), int(end_year) + 1))
