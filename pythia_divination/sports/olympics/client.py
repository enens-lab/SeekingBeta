"""Olympics data client: download + aggregate Summer Games medal tables."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from .constants import ATHLETE_EVENTS_URL, DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "sports" / "olympics"


def load_athlete_events() -> pd.DataFrame:
    """Athlete-event rows (cached permanently — the historical set is static)."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_ROOT / "athlete_events.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    resp = requests.get(
        ATHLETE_EVENTS_URL, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=DEFAULT_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(resp.content))


def summer_medal_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Medals per (year, NOC) for the Summer Games.

    A team-event medal appears as many athlete rows; dedup to one medal per
    (Year, Event, Medal, NOC). Returns columns: year, noc, gold, total.
    """
    if df is None:
        df = load_athlete_events()
    summer = df[(df["Season"] == "Summer") & df["Medal"].notna()].copy()
    # one medal per country per event (collapse team events)
    medals = summer.drop_duplicates(subset=["Year", "Event", "Medal", "NOC"])
    grouped = (
        medals.groupby(["Year", "NOC", "Medal"]).size().unstack(fill_value=0).reset_index()
    )
    for col in ("Gold", "Silver", "Bronze"):
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["total"] = grouped["Gold"] + grouped["Silver"] + grouped["Bronze"]
    grouped = grouped.rename(columns={"Year": "year", "NOC": "noc", "Gold": "gold"})
    return grouped[["year", "noc", "gold", "total"]].sort_values(["year", "total"], ascending=[True, False]).reset_index(drop=True)
