"""Static MLB constants used by ingestion and feature prep."""

from __future__ import annotations

MLB_STATS_API_BASE_URL = "https://statsapi.mlb.com"
BASEBALL_SAVANT_BASE_URL = "https://baseballsavant.mlb.com"

MLB_SCHEDULE_PATH = "/api/v1/schedule"
MLB_TEAMS_PATH = "/api/v1/teams"
MLB_PEOPLE_PATH = "/api/v1/people/{person_id}"
MLB_TEAM_STATS_PATH = "/api/v1/teams/{team_id}/stats"
MLB_PLAYER_STATS_PATH = "/api/v1/people/{person_id}/stats"
MLB_GAME_FEED_PATH = "/api/v1.1/game/{game_pk}/feed/live"
MLB_GAME_BOXSCORE_PATH = "/api/v1/game/{game_pk}/boxscore"
MLB_GAME_PLAY_BY_PLAY_PATH = "/api/v1/game/{game_pk}/playByPlay"
MLB_GAME_CONTEXT_METRICS_PATH = "/api/v1/game/{game_pk}/contextMetrics"
MLB_GAME_WIN_PROBABILITY_PATH = "/api/v1/game/{game_pk}/winProbability"

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)

DEFAULT_SPORT_ID = 1
DEFAULT_REGULAR_SEASON_GAME_TYPES = ("R",)
DEFAULT_TEAM_STAT_GROUPS = ("hitting", "pitching", "fielding")
DEFAULT_TEAM_STAT_TYPES = ("season", "seasonAdvanced")
DEFAULT_PLAYER_SPLIT_CODES = (
    "vl",  # vs left-handed pitching
    "vr",  # vs right-handed pitching
    "h",   # home
    "a",   # away
)
