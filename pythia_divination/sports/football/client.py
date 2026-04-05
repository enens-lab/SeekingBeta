"""nflverse-backed client helpers for Football modeling."""

from __future__ import annotations

from io import StringIO
import logging
from typing import Iterable

import pandas as pd
import requests
from requests import Session
from requests.exceptions import SSLError

from .constants import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    PLAYERS_URL,
    PLAYER_WEEK_STATS_TEMPLATE,
    SCHEDULES_URL,
    TEAM_WEEK_STATS_TEMPLATE,
    WEEKLY_ROSTERS_TEMPLATE,
)

logger = logging.getLogger(__name__)


class FootballStatsClient:
    """Small wrapper around nflverse release assets published on GitHub."""

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        session: Session | None = None,
        allow_insecure_ssl_fallback: bool = True,
    ) -> None:
        self.timeout = timeout
        self.allow_insecure_ssl_fallback = allow_insecure_ssl_fallback
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "text/csv,application/octet-stream,*/*",
                "User-Agent": user_agent,
            }
        )

    def _get_text(self, url: str) -> str:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except SSLError:
            if not self.allow_insecure_ssl_fallback:
                raise
            logger.warning("SSL verification failed for %s; retrying without certificate verification", url)
            response = self.session.get(url, timeout=self.timeout, verify=False)
            response.raise_for_status()
            return response.text

    def _read_csv(self, url: str, *, usecols: Iterable[str] | None = None) -> pd.DataFrame:
        text = self._get_text(url)
        return pd.read_csv(StringIO(text), low_memory=False, usecols=usecols)

    def get_games(self) -> pd.DataFrame:
        return self._read_csv(SCHEDULES_URL)

    def get_team_week_stats(self, season: int) -> pd.DataFrame:
        return self._read_csv(TEAM_WEEK_STATS_TEMPLATE.format(season=int(season)))

    def get_player_week_stats(self, season: int) -> pd.DataFrame:
        return self._read_csv(PLAYER_WEEK_STATS_TEMPLATE.format(season=int(season)))

    def get_weekly_rosters(self, season: int) -> pd.DataFrame:
        return self._read_csv(WEEKLY_ROSTERS_TEMPLATE.format(season=int(season)))

    def get_players(self) -> pd.DataFrame:
        return self._read_csv(PLAYERS_URL)
