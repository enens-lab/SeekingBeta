"""Feature engineering helpers for MLB matchup modeling."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

_WIND_PATTERN = re.compile(r"(?P<mph>\\d+)\\s*mph(?:,\\s*(?P<direction>.*))?$", re.IGNORECASE)
_SAFE_TEAM_LOG_BASE_COLUMNS = {"days_rest", "games_played_prior", "win_pct_prior", "same_site_win_pct_last_10"}
_SAFE_STARTER_LOG_BASE_COLUMNS = {"days_rest", "starts_prior", "starter_win_pct_prior"}


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0 or math.isnan(num) or math.isnan(den):
        return None
    return num / den


def _parse_wind(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    match = _WIND_PATTERN.search(text)
    if not match:
        return None, text
    mph = _safe_float(match.group("mph"))
    direction = match.group("direction")
    return mph, direction.strip() if direction else None


def prepare_games(
    schedule_df: pd.DataFrame,
    details_df: pd.DataFrame,
    *,
    team_meta_df: pd.DataFrame | None = None,
    pitcher_profiles_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge normalized schedule/detail tables into one game frame."""
    games = schedule_df.copy()
    games["official_date"] = pd.to_datetime(games["official_date"], utc=True, errors="coerce").dt.tz_localize(None)
    games = games.loc[games["winner_team_id"].notna()].copy()
    merged = games.merge(details_df, on="game_pk", how="left", suffixes=("", "_detail"))

    merged["home_win"] = (merged["winner_team_id"] == merged["home_team_id"]).astype(int)
    merged["away_win"] = 1 - merged["home_win"]
    merged["month"] = merged["official_date"].dt.month
    merged["day_of_week"] = merged["official_date"].dt.dayofweek
    merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)

    wind = merged["weather_wind"].map(_parse_wind)
    merged["weather_wind_mph"] = wind.map(lambda item: item[0])
    merged["weather_wind_direction"] = wind.map(lambda item: item[1])
    merged["venue_dimensions_sum"] = merged[
        [
            "venue_left_line",
            "venue_left",
            "venue_left_center",
            "venue_center",
            "venue_right_center",
            "venue_right_line",
        ]
    ].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    merged["park_asymmetry"] = (
        pd.to_numeric(merged["venue_left_line"], errors="coerce")
        - pd.to_numeric(merged["venue_right_line"], errors="coerce")
    ).abs()

    if team_meta_df is not None and not team_meta_df.empty:
        team_meta = team_meta_df.rename(columns={"team_id": "away_team_id"})
        away_cols = {
            "abbreviation": "away_team_abbreviation",
            "division_name": "away_division_name",
            "league_name": "away_league_name",
        }
        merged = merged.merge(team_meta[list({"away_team_id", *away_cols.keys()})], on="away_team_id", how="left")
        merged = merged.rename(columns=away_cols)

        team_meta = team_meta_df.rename(columns={"team_id": "home_team_id"})
        home_cols = {
            "abbreviation": "home_team_abbreviation",
            "division_name": "home_division_name",
            "league_name": "home_league_name",
        }
        merged = merged.merge(team_meta[list({"home_team_id", *home_cols.keys()})], on="home_team_id", how="left")
        merged = merged.rename(columns=home_cols)

    if pitcher_profiles_df is not None and not pitcher_profiles_df.empty:
        away_profiles = pitcher_profiles_df.rename(columns={"player_id": "away_probable_pitcher_id"}).add_prefix("away_starter_profile_")
        away_profiles = away_profiles.rename(columns={"away_starter_profile_away_probable_pitcher_id": "away_probable_pitcher_id"})
        merged = merged.merge(away_profiles, on="away_probable_pitcher_id", how="left")

        home_profiles = pitcher_profiles_df.rename(columns={"player_id": "home_probable_pitcher_id"}).add_prefix("home_starter_profile_")
        home_profiles = home_profiles.rename(columns={"home_starter_profile_home_probable_pitcher_id": "home_probable_pitcher_id"})
        merged = merged.merge(home_profiles, on="home_probable_pitcher_id", how="left")

    merged = merged.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
    return merged


def build_team_game_logs(games: pd.DataFrame) -> pd.DataFrame:
    """Convert matchup rows into one row per team-game for rolling history."""
    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for side, opp in (("away", "home"), ("home", "away")):
            rows.append(
                {
                    "game_pk": game["game_pk"],
                    "official_date": game["official_date"],
                    "season": game.get("season"),
                    "team_side": side,
                    "team_id": game[f"{side}_team_id"],
                    "team_name": game[f"{side}_team_name"],
                    "team_abbreviation": game.get(f"{side}_team_abbreviation"),
                    "opponent_team_id": game[f"{opp}_team_id"],
                    "is_home": 1 if side == "home" else 0,
                    "won": game[f"{side}_is_winner"],
                    "runs_scored": game[f"{side}_score"],
                    "runs_allowed": game[f"{opp}_score"],
                    "run_diff": (game[f"{side}_score"] or 0) - (game[f"{opp}_score"] or 0),
                    "starter_id": game.get(f"{side}_probable_pitcher_id"),
                    "starter_name": game.get(f"{side}_probable_pitcher_name"),
                    "batting_hits": game.get(f"{side}_batting_hits"),
                    "batting_home_runs": game.get(f"{side}_batting_home_runs"),
                    "batting_walks": game.get(f"{side}_batting_walks"),
                    "batting_strikeouts": game.get(f"{side}_batting_strikeouts"),
                    "batting_total_bases": game.get(f"{side}_batting_total_bases"),
                    "batting_ops": game.get(f"{side}_batting_ops"),
                    "pitching_hits_allowed": game.get(f"{side}_pitching_hits_allowed"),
                    "pitching_earned_runs": game.get(f"{side}_pitching_earned_runs"),
                    "pitching_walks": game.get(f"{side}_pitching_walks"),
                    "pitching_strikeouts": game.get(f"{side}_pitching_strikeouts"),
                    "pitching_home_runs_allowed": game.get(f"{side}_pitching_home_runs_allowed"),
                    "pitching_whip": game.get(f"{side}_pitching_whip"),
                    "pitching_pitches_thrown": game.get(f"{side}_pitching_pitches_thrown"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["team_id", "official_date", "game_pk"]).reset_index(drop=True)
    frame["official_date"] = pd.to_datetime(frame["official_date"], errors="coerce")
    frame["days_rest"] = frame.groupby("team_id")["official_date"].diff().dt.days

    metrics = [
        "won",
        "runs_scored",
        "runs_allowed",
        "run_diff",
        "batting_hits",
        "batting_home_runs",
        "batting_walks",
        "batting_strikeouts",
        "batting_total_bases",
        "batting_ops",
        "pitching_hits_allowed",
        "pitching_earned_runs",
        "pitching_walks",
        "pitching_strikeouts",
        "pitching_home_runs_allowed",
        "pitching_whip",
        "pitching_pitches_thrown",
    ]
    windows = (3, 5, 10)
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        grouped = frame.groupby("team_id")[metric]
        for window in windows:
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    frame["games_played_prior"] = frame.groupby("team_id").cumcount()
    frame["win_pct_prior"] = frame.groupby("team_id")["won"].transform(lambda s: s.shift(1).expanding().mean())
    frame["same_site_win_pct_last_10"] = frame.groupby(["team_id", "is_home"])["won"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    return frame


def build_starter_game_logs(games: pd.DataFrame) -> pd.DataFrame:
    """Convert matchup rows into one row per starting pitcher appearance."""
    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for side in ("away", "home"):
            starter_id = game.get(f"{side}_probable_pitcher_id")
            if starter_id is None:
                continue
            innings = pd.to_numeric(pd.Series([game.get(f"{side}_starter_innings_pitched")]), errors="coerce").iloc[0]
            hits_allowed = pd.to_numeric(pd.Series([game.get(f"{side}_starter_hits_allowed")]), errors="coerce").iloc[0]
            walks = pd.to_numeric(pd.Series([game.get(f"{side}_starter_walks")]), errors="coerce").iloc[0]
            strikeouts = pd.to_numeric(pd.Series([game.get(f"{side}_starter_strikeouts")]), errors="coerce").iloc[0]
            earned_runs = pd.to_numeric(pd.Series([game.get(f"{side}_starter_earned_runs")]), errors="coerce").iloc[0]
            home_runs_allowed = pd.to_numeric(pd.Series([game.get(f"{side}_starter_home_runs_allowed")]), errors="coerce").iloc[0]
            pitches = pd.to_numeric(pd.Series([game.get(f"{side}_starter_pitches_thrown")]), errors="coerce").iloc[0]
            strikes = pd.to_numeric(pd.Series([game.get(f"{side}_starter_strikes")]), errors="coerce").iloc[0]
            whip = _safe_divide((hits_allowed or 0) + (walks or 0), innings)
            era = _safe_divide((earned_runs or 0) * 9.0, innings)
            strike_pct = _safe_divide(strikes, pitches)
            rows.append(
                {
                    "game_pk": game["game_pk"],
                    "official_date": game["official_date"],
                    "season": game.get("season"),
                    "starter_id": starter_id,
                    "starter_name": game.get(f"{side}_probable_pitcher_name"),
                    "team_id": game.get(f"{side}_team_id"),
                    "team_name": game.get(f"{side}_team_name"),
                    "team_side": side,
                    "won": game.get(f"{side}_is_winner"),
                    "innings_pitched": innings,
                    "earned_runs": earned_runs,
                    "hits_allowed": hits_allowed,
                    "walks": walks,
                    "strikeouts": strikeouts,
                    "home_runs_allowed": home_runs_allowed,
                    "pitches_thrown": pitches,
                    "strikes": strikes,
                    "batters_faced": pd.to_numeric(pd.Series([game.get(f"{side}_starter_batters_faced")]), errors="coerce").iloc[0],
                    "whip": whip,
                    "era_like": era,
                    "strike_pct": strike_pct,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["starter_id", "official_date", "game_pk"]).reset_index(drop=True)
    frame["official_date"] = pd.to_datetime(frame["official_date"], errors="coerce")
    frame["days_rest"] = frame.groupby("starter_id")["official_date"].diff().dt.days

    metrics = [
        "won",
        "innings_pitched",
        "earned_runs",
        "hits_allowed",
        "walks",
        "strikeouts",
        "home_runs_allowed",
        "pitches_thrown",
        "batters_faced",
        "whip",
        "era_like",
        "strike_pct",
    ]
    windows = (3, 5)
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        grouped = frame.groupby("starter_id")[metric]
        for window in windows:
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )
    frame["starts_prior"] = frame.groupby("starter_id").cumcount()
    frame["starter_win_pct_prior"] = frame.groupby("starter_id")["won"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return frame


def merge_team_features(games: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    if team_logs.empty:
        return games
    safe_feature_columns = {
        column
        for column in team_logs.columns
        if column.endswith("_avg_last_3")
        or column.endswith("_avg_last_5")
        or column.endswith("_avg_last_10")
        or column in _SAFE_TEAM_LOG_BASE_COLUMNS
    }
    away_cols = {
        column: f"away_team_{column}"
        for column in team_logs.columns
        if column in safe_feature_columns
    }
    away = team_logs.loc[team_logs["team_side"] == "away", ["game_pk", "team_id", *away_cols.keys()]].rename(columns=away_cols)
    away = away.rename(columns={"team_id": "away_team_id"})

    home_cols = {
        column: f"home_team_{column}"
        for column in team_logs.columns
        if column in safe_feature_columns
    }
    home = team_logs.loc[team_logs["team_side"] == "home", ["game_pk", "team_id", *home_cols.keys()]].rename(columns=home_cols)
    home = home.rename(columns={"team_id": "home_team_id"})

    merged = games.merge(away, on=["game_pk", "away_team_id"], how="left")
    merged = merged.merge(home, on=["game_pk", "home_team_id"], how="left")
    return merged


def merge_starter_features(games: pd.DataFrame, starter_logs: pd.DataFrame) -> pd.DataFrame:
    if starter_logs.empty:
        return games
    safe_feature_columns = {
        column
        for column in starter_logs.columns
        if column.endswith("_avg_last_3")
        or column.endswith("_avg_last_5")
        or column in _SAFE_STARTER_LOG_BASE_COLUMNS
    }
    away_cols = {
        column: f"away_starter_{column}"
        for column in starter_logs.columns
        if column in safe_feature_columns
    }
    away = starter_logs.loc[starter_logs["team_side"] == "away", ["game_pk", "starter_id", "team_id", *away_cols.keys()]].rename(columns=away_cols)
    away = away.rename(columns={"starter_id": "away_probable_pitcher_id", "team_id": "away_team_id"})

    home_cols = {
        column: f"home_starter_{column}"
        for column in starter_logs.columns
        if column in safe_feature_columns
    }
    home = starter_logs.loc[starter_logs["team_side"] == "home", ["game_pk", "starter_id", "team_id", *home_cols.keys()]].rename(columns=home_cols)
    home = home.rename(columns={"starter_id": "home_probable_pitcher_id", "team_id": "home_team_id"})

    merged = games.merge(away, on=["game_pk", "away_probable_pitcher_id", "away_team_id"], how="left")
    merged = merged.merge(home, on=["game_pk", "home_probable_pitcher_id", "home_team_id"], how="left")
    return merged


def add_matchup_differentials(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    candidate_pairs = [
        ("team_win_pct_prior", "win_pct_prior"),
        ("team_runs_scored_avg_last_5", "runs_scored_avg_last_5"),
        ("team_runs_allowed_avg_last_5", "runs_allowed_avg_last_5"),
        ("team_run_diff_avg_last_10", "run_diff_avg_last_10"),
        ("starter_era_like_avg_last_5", "era_like_avg_last_5"),
        ("starter_whip_avg_last_5", "whip_avg_last_5"),
        ("starter_strikeouts_avg_last_5", "strikeouts_avg_last_5"),
        ("starter_walks_avg_last_5", "walks_avg_last_5"),
        ("starter_days_rest", "days_rest"),
    ]
    for left_suffix, right_suffix in candidate_pairs:
        away_column = f"away_{left_suffix}"
        home_column = f"home_{left_suffix}"
        if away_column in frame.columns and home_column in frame.columns:
            frame[f"delta_{right_suffix}"] = pd.to_numeric(frame["home_" + left_suffix], errors="coerce") - pd.to_numeric(
                frame["away_" + left_suffix],
                errors="coerce",
            )
            continue
        away_column = f"away_team_{right_suffix}"
        home_column = f"home_team_{right_suffix}"
        if away_column in frame.columns and home_column in frame.columns:
            frame[f"delta_{right_suffix}"] = pd.to_numeric(frame[home_column], errors="coerce") - pd.to_numeric(
                frame[away_column],
                errors="coerce",
            )
            continue
        away_column = f"away_starter_{right_suffix}"
        home_column = f"home_starter_{right_suffix}"
        if away_column in frame.columns and home_column in frame.columns:
            frame[f"delta_{right_suffix}"] = pd.to_numeric(frame[home_column], errors="coerce") - pd.to_numeric(
                frame[away_column],
                errors="coerce",
            )

    frame["home_field_flag"] = 1
    frame["temperature_f"] = pd.to_numeric(frame["weather_temp_f"], errors="coerce")
    return frame
