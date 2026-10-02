"""Feature engineering for ATP/WTA tennis prediction models.

Ratings come from ``sports.wta.elo`` (stable player identity across the 2025 ATP
id-scheme change, FiveThirtyEight-style decaying K, overall/surface blend). All
per-match and per-event numbers are pre-match: they use only earlier matches.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from sports.wta.elo import (
    display_names,
    load_params,
    normalize_identifier,
    prepare_matches,
    run_elo,
)

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 10
DEFAULT_SERVE_WON = 0.60
DEFAULT_RETURN_WON = 0.40


def calculate_elo_updates(
    winner_elo: float,
    loser_elo: float,
    k: float = 32,
) -> tuple[float, float]:
    """Single fixed-K Elo update (kept for callers outside the tennis pipeline)."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner
    return winner_elo + k * (1 - expected_winner), loser_elo + k * (0 - expected_loser)


def _to_float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return _to_float_array(frame, column)


def _serve_return_shares(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-match serve-points-won and return-points-won shares for winner and loser."""
    w_svpt = _to_float_array(frame, "w_svpt")
    l_svpt = _to_float_array(frame, "l_svpt")
    w_won = _to_float_array(frame, "w_1stWon") + _to_float_array(frame, "w_2ndWon")
    l_won = _to_float_array(frame, "l_1stWon") + _to_float_array(frame, "l_2ndWon")
    with np.errstate(divide="ignore", invalid="ignore"):
        w_serve = np.where(w_svpt > 0, w_won / w_svpt, DEFAULT_SERVE_WON)
        l_serve = np.where(l_svpt > 0, l_won / l_svpt, DEFAULT_SERVE_WON)
        w_return = np.where(l_svpt > 0, (l_svpt - l_won) / l_svpt, DEFAULT_RETURN_WON)
        l_return = np.where(w_svpt > 0, (w_svpt - w_won) / w_svpt, DEFAULT_RETURN_WON)
    w_serve = np.where(np.isfinite(w_serve), w_serve, DEFAULT_SERVE_WON)
    l_serve = np.where(np.isfinite(l_serve), l_serve, DEFAULT_SERVE_WON)
    w_return = np.where(np.isfinite(w_return), w_return, DEFAULT_RETURN_WON)
    l_return = np.where(np.isfinite(l_return), l_return, DEFAULT_RETURN_WON)
    return w_serve, w_return, l_serve, l_return


def process_match_history(matches: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Chronological pass: stable player keys, pre-match Elo and rolling serve/return form.

    Output keeps the original match columns plus ``winner_key``/``loser_key``,
    ``w_pre_match_elo``/``l_pre_match_elo`` (overall), ``*_surf_elo``,
    ``*_blend_elo`` (what the match probability uses), ``p_winner_elo`` and
    ``w_/l_rolling_serve_won``/``return_won`` (mean of the previous 10 matches).
    """
    params = params or load_params()
    prepared = prepare_matches(matches, params)
    values, _, _ = run_elo(prepared, params)
    df = pd.concat([prepared, values], axis=1)
    df["winner_id"] = df["winner_id"].map(normalize_identifier)
    df["loser_id"] = df["loser_id"].map(normalize_identifier)
    df["w_pre_match_elo"] = df["w_elo"]
    df["l_pre_match_elo"] = df["l_elo"]
    df["w_pre_match_surf_elo"] = df["w_surf_elo"]
    df["l_pre_match_surf_elo"] = df["l_surf_elo"]
    df["w_pre_match_blend_elo"] = df["w_blend_elo"]
    df["l_pre_match_blend_elo"] = df["l_blend_elo"]
    df["p_winner_elo"] = df["p_winner"]

    w_serve, w_return, l_serve, l_return = _serve_return_shares(df)
    history: dict[str, list[tuple[float, float]]] = {}
    out = {name: np.empty(len(df)) for name in ("w_serve", "w_return", "l_serve", "l_return")}

    def rolling(key: str) -> tuple[float, float]:
        past = history.get(key)
        if not past:
            return 0.5, 0.5
        recent = past[-ROLLING_WINDOW:]
        return float(np.mean([item[0] for item in recent])), float(np.mean([item[1] for item in recent]))

    for index, (winner, loser) in enumerate(zip(df["winner_key"].values, df["loser_key"].values)):
        out["w_serve"][index], out["w_return"][index] = rolling(winner)
        out["l_serve"][index], out["l_return"][index] = rolling(loser)
        history.setdefault(winner, []).append((w_serve[index], w_return[index]))
        history.setdefault(loser, []).append((l_serve[index], l_return[index]))

    df["w_serve_share"] = w_serve
    df["w_return_share"] = w_return
    df["l_serve_share"] = l_serve
    df["l_return_share"] = l_return
    df["w_rolling_serve_won"] = out["w_serve"]
    df["w_rolling_return_won"] = out["w_return"]
    df["l_rolling_serve_won"] = out["l_serve"]
    df["l_rolling_return_won"] = out["l_return"]
    return df


def event_champions(history: pd.DataFrame) -> dict[str, str]:
    """Champion per event: the winner of the final (round == "F").

    Events without round labels (tennis-data conversions) fall back to the one
    player with at least one win and no loss in the event; anything else (round
    robins without a final, incomplete data) has no champion and is never graded.
    """
    champions: dict[str, str] = {}
    rounds = history.get("round", pd.Series("", index=history.index)).astype(str).str.strip().str.upper()
    finals = history[rounds == "F"]
    for event_id, group in finals.groupby("event_id", sort=False):
        champions[event_id] = str(group["winner_key"].iloc[-1])

    remaining = history[~history["event_id"].isin(champions.keys())]
    for event_id, group in remaining.groupby("event_id", sort=False):
        if (rounds.loc[group.index] == "RR").all():
            continue
        winners = set(group["winner_key"])
        losers = set(group["loser_key"])
        unbeaten = winners - losers
        if len(unbeaten) == 1:
            champions[event_id] = next(iter(unbeaten))
    return champions


def build_player_event_features(match_history: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, player): pre-event ratings, form, rank and the label.

    Team competitions (Davis Cup, BJK Cup, United Cup, Laver Cup, ...) are
    dropped: their "events" are ties or group stages, not individual fields.
    """
    history = match_history[~match_history["is_team_event"]].copy()
    if history.empty:
        return pd.DataFrame()
    names = display_names(history)
    champions = event_champions(history)

    def side_frame(prefix: str, short: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "tournament_id": history["event_id"].values,
                "source_tournament_id": history["tourney_id"].astype(str).values,
                "tournament_name": history["tourney_name"].values,
                "surface": history["surface"].values if "surface" in history.columns else "Unknown",
                "level": history["tourney_level"].values if "tourney_level" in history.columns else "",
                "match_date": history["tourney_date"].values,
                "player_key": history[f"{prefix}_key"].values,
                "player_id": history[f"{prefix}_id"].values,
                "raw_name": history[f"{prefix}_name"].values,
                "elo": history[f"{short}_pre_match_elo"].values,
                "surf_elo": history[f"{short}_pre_match_surf_elo"].values,
                "blend_elo": history[f"{short}_pre_match_blend_elo"].values,
                "serve_won": history[f"{short}_rolling_serve_won"].values,
                "return_won": history[f"{short}_rolling_return_won"].values,
                "age": _numeric_column(history, f"{prefix}_age"),
                "height": _numeric_column(history, f"{prefix}_ht"),
                "rank": _numeric_column(history, f"{prefix}_rank"),
                # Draw seed; ESPN-sourced seasons have seeds but no rankings.
                "seed": _numeric_column(history, f"{prefix}_seed"),
                "matches_played": history[f"{short}_matches"].values,
                "tour": history["tour"].values,
                "_order": np.arange(len(history)),
            }
        )
        return frame

    events = pd.concat([side_frame("winner", "w"), side_frame("loser", "l")], ignore_index=True)
    # A player's first match in an event carries their pre-event numbers.
    events = events.sort_values("_order", kind="mergesort")
    bounds = events.groupby("tournament_id")["match_date"].agg(["min", "max"])
    events = events.drop_duplicates(subset=["tournament_id", "player_key"], keep="first").copy()
    events["date"] = events["tournament_id"].map(bounds["min"]).astype(int)
    events["end_date"] = events["tournament_id"].map(bounds["max"]).astype(int)
    events["player_name"] = [names.get(key, raw) for key, raw in zip(events["player_key"], events["raw_name"])]
    events["won_tournament"] = [
        1 if champions.get(event_id) == key else 0
        for event_id, key in zip(events["tournament_id"], events["player_key"])
    ]
    events["has_champion"] = events["tournament_id"].isin(champions.keys())
    events["field_size"] = events.groupby("tournament_id")["player_key"].transform("count")
    events["elo_field_percentile"] = events.groupby("tournament_id")["blend_elo"].rank(pct=True)
    return events.drop(columns=["_order", "raw_name", "match_date"]).reset_index(drop=True)


def add_rolling_features(player_events: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and form features without leaking across tours."""
    df = player_events.sort_values(["player_key", "date", "tournament_id"]).copy()
    grouped = df.groupby("player_key", sort=False)

    df["player_rolling_win_rate_5"] = grouped["won_tournament"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean(),
    )
    df["player_elo_diff_5"] = grouped["elo"].transform(lambda values: values - values.shift(5))

    return df
