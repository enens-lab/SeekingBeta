"""Optional DataGolf API client for richer PGA feature collection.

This client is intentionally thin and query-string based so we can plug in
additional feeds without coupling the rest of the sports pipeline to a single
provider. A DataGolf API key is required for all requests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATAGOLF_BASE_URL = "https://feeds.datagolf.com"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)


class DataGolfClient:
    """Small API wrapper around the official DataGolf feed endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DATAGOLF_BASE_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DATAGOLF_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError(
                "DATAGOLF_API_KEY is required for DataGolf ingestion. "
                "Get a key from https://datagolf.com/api-access."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "application/json,text/csv,*/*",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

    def _request(self, endpoint: str, **params: Any) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        query = {k: v for k, v in params.items() if v not in (None, "", [], ())}
        query["key"] = self.api_key
        response = self.session.get(url, params=query, timeout=self.timeout_seconds)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "json" in content_type.lower() or response.text.lstrip().startswith(("{", "[")):
            return response.json()
        return response.text

    @staticmethod
    def to_frame(payload: Any) -> pd.DataFrame:
        if payload is None:
            return pd.DataFrame()
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            list_keys = [key for key, value in payload.items() if isinstance(value, list)]
            if len(list_keys) == 1:
                frame = pd.json_normalize(payload[list_keys[0]])
                if frame.empty:
                    return pd.DataFrame([payload])
                return frame
            return pd.json_normalize(payload)
        raise TypeError(f"Unsupported payload type for DataFrame conversion: {type(payload)!r}")

    def get_player_list(self, *, file_format: str = "json") -> Any:
        return self._request("get-player-list", file_format=file_format)

    def get_schedule(
        self,
        *,
        tour: str = "pga",
        season: int | None = None,
        upcoming_only: str = "no",
        file_format: str = "json",
    ) -> Any:
        return self._request(
            "get-schedule",
            tour=tour,
            season=season,
            upcoming_only=upcoming_only,
            file_format=file_format,
        )

    def get_field_updates(self, *, tour: str = "pga", file_format: str = "json") -> Any:
        return self._request("field-updates", tour=tour, file_format=file_format)

    def get_pre_tournament_predictions(
        self,
        *,
        tour: str = "pga",
        add_position: str | None = None,
        dead_heat: str = "yes",
        odds_format: str = "percent",
        file_format: str = "json",
    ) -> Any:
        return self._request(
            "preds/pre-tournament",
            tour=tour,
            add_position=add_position,
            dead_heat=dead_heat,
            odds_format=odds_format,
            file_format=file_format,
        )

    def get_pre_tournament_archive(
        self,
        *,
        event_id: str,
        year: int,
        odds_format: str = "percent",
        file_format: str = "json",
    ) -> Any:
        return self._request(
            "preds/pre-tournament-archive",
            event_id=event_id,
            year=year,
            odds_format=odds_format,
            file_format=file_format,
        )

    def get_player_decompositions(self, *, tour: str = "pga", file_format: str = "json") -> Any:
        return self._request("preds/player-decompositions", tour=tour, file_format=file_format)

    def get_skill_ratings(self, *, display: str = "value", file_format: str = "json") -> Any:
        return self._request("preds/skill-ratings", display=display, file_format=file_format)

    def get_approach_skill(self, *, period: str = "l24", file_format: str = "json") -> Any:
        return self._request("preds/approach-skill", period=period, file_format=file_format)

    def get_live_tournament_stats(
        self,
        *,
        stats: str | None = None,
        round_value: str = "event_avg",
        display: str = "value",
        file_format: str = "json",
    ) -> Any:
        return self._request(
            "preds/live-tournament-stats",
            stats=stats,
            round=round_value,
            display=display,
            file_format=file_format,
        )

    def get_historical_raw_event_list(self, *, tour: str = "pga", file_format: str = "json") -> Any:
        return self._request("historical-raw-data/event-list", tour=tour, file_format=file_format)

    def get_historical_raw_rounds(
        self,
        *,
        tour: str,
        event_id: str,
        year: int,
        file_format: str = "json",
    ) -> Any:
        return self._request(
            "historical-raw-data/rounds",
            tour=tour,
            event_id=event_id,
            year=year,
            file_format=file_format,
        )

    def get_historical_event_list(self, *, tour: str = "pga", file_format: str = "json") -> Any:
        return self._request("historical-event-data/event-list", tour=tour, file_format=file_format)

    def get_historical_event_data(
        self,
        *,
        tour: str,
        event_id: str,
        year: int,
        file_format: str = "json",
    ) -> Any:
        return self._request(
            "historical-event-data/events",
            tour=tour,
            event_id=event_id,
            year=year,
            file_format=file_format,
        )


def normalize_datagolf_columns(frame: pd.DataFrame, *, prefix: str | None = None) -> pd.DataFrame:
    """Normalize DataGolf columns to a merge-friendly snake_case shape."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    renamed: dict[str, str] = {}
    for column in normalized.columns:
        clean = (
            str(column)
            .strip()
            .lower()
            .replace("%", " pct ")
            .replace("/", " ")
            .replace("-", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace(".", " ")
        )
        clean = "_".join(part for part in clean.split() if part)
        clean = clean or "value"
        renamed[column] = f"{prefix}{clean}" if prefix else clean
    return normalized.rename(columns=renamed)
