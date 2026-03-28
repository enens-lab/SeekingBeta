"""Feature engineering for ATP/WTA tennis prediction models."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EloConfig:
    k_factor: float = 32
    initial_elo: float = 1500.0


def calculate_elo_updates(
    winner_elo: float,
    loser_elo: float,
    k: float = 32,
) -> tuple[float, float]:
    """Calculate the new Elo ratings after a match."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner

    new_winner_elo = winner_elo + k * (1 - expected_winner)
    new_loser_elo = loser_elo + k * (0 - expected_loser)

    return new_winner_elo, new_loser_elo


def _normalize_identifier(value: Any) -> str | None:
    if pd.isna(value):
        return None

    identifier = str(value).strip()
    if not identifier or identifier.lower() in {"nan", "none"}:
        return None
    if identifier.endswith(".0") and identifier[:-2].isdigit():
        return identifier[:-2]
    return identifier


def _to_float(value: Any, default: float) -> float:
    if pd.isna(value) or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_tournament_id(tour: str, raw_tourney_id: str | None) -> str | None:
    tournament_id = _normalize_identifier(raw_tourney_id)
    if tournament_id is None:
        return None
    return f"{tour}:{tournament_id}"


def _canonical_player_key(tour: str, raw_player_id: str | None) -> str | None:
    player_id = _normalize_identifier(raw_player_id)
    if player_id is None:
        return None
    return f"{tour}:{player_id}"


def process_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Iterate through matches chronologically to calculate rolling stats and Elo."""
    df = matches.sort_values(["tourney_date", "tour", "tourney_id", "match_num"]).copy()

    player_elos: dict[str, dict[str, float]] = {}
    surface_elos: dict[str, dict[str, dict[str, float]]] = {}
    player_stats: dict[tuple[str, str], list[dict[str, float]]] = {}

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        try:
            tour = str(row.get("tour", "")).strip().upper()
            if not tour:
                continue

            winner_id = _normalize_identifier(row.get("winner_id"))
            loser_id = _normalize_identifier(row.get("loser_id"))
            if winner_id is None or loser_id is None:
                continue

            surface = str(row.get("surface", "Unknown") or "Unknown").strip() or "Unknown"

            player_elos.setdefault(tour, {})
            surface_elos.setdefault(tour, {})
            surface_elos[tour].setdefault(surface, {})

            winner_key = (tour, winner_id)
            loser_key = (tour, loser_id)

            winner_elo = player_elos[tour].get(winner_id, 1500.0)
            loser_elo = player_elos[tour].get(loser_id, 1500.0)
            winner_surface_elo = surface_elos[tour][surface].get(winner_id, 1500.0)
            loser_surface_elo = surface_elos[tour][surface].get(loser_id, 1500.0)

            winner_svpt = _to_float(row.get("w_svpt"), 0.0)
            loser_svpt = _to_float(row.get("l_svpt"), 0.0)
            winner_first_won = _to_float(row.get("w_1stWon"), 0.0)
            winner_second_won = _to_float(row.get("w_2ndWon"), 0.0)
            loser_first_won = _to_float(row.get("l_1stWon"), 0.0)
            loser_second_won = _to_float(row.get("l_2ndWon"), 0.0)

            if winner_svpt > 0:
                winner_serve_won = (winner_first_won + winner_second_won) / winner_svpt
            else:
                winner_serve_won = 0.60

            if loser_svpt > 0:
                winner_return_won = (loser_svpt - (loser_first_won + loser_second_won)) / loser_svpt
            else:
                winner_return_won = 0.40

            def get_rolling_avg(player_state_key: tuple[str, str], metric: str, window: int = 10) -> float:
                history = player_stats.get(player_state_key, [])
                if not history:
                    return 0.5
                values = [entry[metric] for entry in history[-window:]]
                return float(sum(values) / len(values))

            match_features = row.to_dict()
            match_features["winner_id"] = winner_id
            match_features["loser_id"] = loser_id
            match_features["tour"] = tour
            match_features["w_pre_match_elo"] = winner_elo
            match_features["l_pre_match_elo"] = loser_elo
            match_features["w_pre_match_surf_elo"] = winner_surface_elo
            match_features["l_pre_match_surf_elo"] = loser_surface_elo
            match_features["w_rolling_serve_won"] = get_rolling_avg(winner_key, "serve_won")
            match_features["w_rolling_return_won"] = get_rolling_avg(winner_key, "return_won")
            match_features["l_rolling_serve_won"] = get_rolling_avg(loser_key, "serve_won")
            match_features["l_rolling_return_won"] = get_rolling_avg(loser_key, "return_won")
            rows.append(match_features)

            new_winner_elo, new_loser_elo = calculate_elo_updates(winner_elo, loser_elo)
            player_elos[tour][winner_id] = new_winner_elo
            player_elos[tour][loser_id] = new_loser_elo

            new_winner_surface_elo, new_loser_surface_elo = calculate_elo_updates(
                winner_surface_elo,
                loser_surface_elo,
                k=16,
            )
            surface_elos[tour][surface][winner_id] = new_winner_surface_elo
            surface_elos[tour][surface][loser_id] = new_loser_surface_elo

            player_stats.setdefault(winner_key, [])
            player_stats.setdefault(loser_key, [])
            player_stats[winner_key].append(
                {"serve_won": winner_serve_won, "return_won": winner_return_won},
            )

            if winner_svpt > 0:
                loser_return_won = (winner_svpt - (winner_first_won + winner_second_won)) / winner_svpt
            else:
                loser_return_won = 0.40
            if loser_svpt > 0:
                loser_serve_won = (loser_first_won + loser_second_won) / loser_svpt
            else:
                loser_serve_won = 0.60

            player_stats[loser_key].append(
                {"serve_won": loser_serve_won, "return_won": loser_return_won},
            )
        except Exception as exc:  # pragma: no cover - defensive path for malformed rows
            logger.debug("Skipping malformed tennis row: %s", exc)
            continue

    return pd.DataFrame(rows)


def _player_event_frame(match_history: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    columns = [
        "tourney_id",
        "tourney_name",
        "surface",
        "tourney_level",
        "tourney_date",
        f"{prefix}_id",
        f"{prefix}_name",
        f"{prefix[0]}_pre_match_elo",
        f"{prefix[0]}_pre_match_surf_elo",
        f"{prefix[0]}_rolling_serve_won",
        f"{prefix[0]}_rolling_return_won",
        f"{prefix}_age",
        f"{prefix}_ht",
        "tour",
    ]
    frame = match_history.reindex(columns=columns).copy()
    frame.columns = [
        "source_tournament_id",
        "tournament_name",
        "surface",
        "level",
        "date",
        "player_id",
        "player_name",
        "elo",
        "surf_elo",
        "serve_won",
        "return_won",
        "age",
        "height",
        "tour",
    ]
    frame["tour"] = frame["tour"].astype(str).str.upper()
    frame["source_tournament_id"] = frame["source_tournament_id"].map(_normalize_identifier)
    frame["player_id"] = frame["player_id"].map(_normalize_identifier)
    frame = frame.dropna(subset=["source_tournament_id", "player_id", "tour"])
    frame["tournament_id"] = frame.apply(
        lambda entry: _canonical_tournament_id(entry["tour"], entry["source_tournament_id"]),
        axis=1,
    )
    frame["player_key"] = frame.apply(
        lambda entry: _canonical_player_key(entry["tour"], entry["player_id"]),
        axis=1,
    )
    return frame


def build_player_event_features(match_history: pd.DataFrame) -> pd.DataFrame:
    """Transform match-level data into player-event-level features."""
    history = match_history.copy()
    history["tour"] = history["tour"].astype(str).str.upper()
    history["source_tournament_id"] = history["tourney_id"].map(_normalize_identifier)
    history["event_id"] = history.apply(
        lambda entry: _canonical_tournament_id(entry["tour"], entry["source_tournament_id"]),
        axis=1,
    )
    history["winner_id"] = history["winner_id"].map(_normalize_identifier)
    history["loser_id"] = history["loser_id"].map(_normalize_identifier)
    history["winner_key"] = history.apply(
        lambda entry: _canonical_player_key(entry["tour"], entry["winner_id"]),
        axis=1,
    )
    history["loser_key"] = history.apply(
        lambda entry: _canonical_player_key(entry["tour"], entry["loser_id"]),
        axis=1,
    )

    winners = _player_event_frame(history, prefix="winner")
    losers = _player_event_frame(history, prefix="loser")

    player_events = pd.concat([winners, losers], ignore_index=True)
    player_events = player_events.drop_duplicates(subset=["tournament_id", "player_key"], keep="first")

    finals = history.sort_values(["tourney_date", "event_id", "match_num"]).groupby("event_id", sort=False).tail(1)
    champions = dict(zip(finals["event_id"], finals["winner_key"]))
    player_events["won_tournament"] = player_events.apply(
        lambda entry: 1 if champions.get(entry["tournament_id"]) == entry["player_key"] else 0,
        axis=1,
    )
    player_events["elo_field_percentile"] = player_events.groupby("tournament_id")["elo"].rank(pct=True)

    return player_events


def add_rolling_features(player_events: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and form features without leaking across tours."""
    df = player_events.sort_values(["player_key", "date", "tournament_id"]).copy()
    grouped = df.groupby("player_key", sort=False)

    df["player_rolling_win_rate_5"] = grouped["won_tournament"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean(),
    )
    df["player_elo_diff_5"] = grouped["elo"].transform(lambda values: values - values.shift(5))

    return df
