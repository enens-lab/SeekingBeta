"""Static Hockey constants for official NHL API ingestion and modeling."""

from __future__ import annotations

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)

API_BASE_URL = "https://api-web.nhle.com/v1"
LEAGUE_KEY = "nhl"
DISPLAY_NAME = "Hockey"
CURRENT_SEASON_START = 2025
DEFAULT_SEASON_START = 2024
DEFAULT_SEASON_END = 2025
REGULAR_SEASON_GAME_TYPE = 2
SUPPORTED_GAME_STATES_COMPLETED = {"OFF", "FINAL"}

TEAM_CODES = (
    "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH",
    "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG",
)


def season_id(start_year: int) -> int:
    return int(f"{int(start_year)}{int(start_year) + 1}")



def season_label(start_year: int) -> str:
    return f"{int(start_year)}-{str(int(start_year) + 1)[-2:]}"



def season_start_years_for_range(start_year: int, end_year: int) -> list[int]:
    if end_year < start_year:
        raise ValueError("season_end must be greater than or equal to season_start")
    return list(range(int(start_year), int(end_year) + 1))
