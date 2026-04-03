"""Feature engineering helpers for basketball matchup modeling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

_SAFE_TEAM_LOG_BASE_COLUMNS = {"days_rest", "games_played_prior", "win_pct_prior", "same_site_win_pct_last_10"}


def _to_datetime_mixed(values: Any) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0 or math.isnan(num) or math.isnan(den):
        return None
    return num / den


def _estimate_possessions(field_goals_attempted: Any, free_throws_attempted: Any, rebounds_offensive: Any, turnovers: Any) -> float | None:
    fga = pd.to_numeric(pd.Series([field_goals_attempted]), errors="coerce").iloc[0]
    fta = pd.to_numeric(pd.Series([free_throws_attempted]), errors="coerce").iloc[0]
    oreb = pd.to_numeric(pd.Series([rebounds_offensive]), errors="coerce").iloc[0]
    tov = pd.to_numeric(pd.Series([turnovers]), errors="coerce").iloc[0]
    if pd.isna(fga) or pd.isna(fta) or pd.isna(oreb) or pd.isna(tov):
        return None
    return float(fga + 0.44 * fta - oreb + tov)


def prepare_games(schedule_df: pd.DataFrame, details_df: pd.DataFrame, *, require_completed: bool = True) -> pd.DataFrame:
    games = schedule_df.copy()
    games["official_date"] = _to_datetime_mixed(games["official_date"])
    merged = games.merge(details_df, on=["league", "game_id"], how="left", suffixes=("", "_detail"))
    if "official_date_detail" in merged.columns:
        merged["official_date"] = merged["official_date"].fillna(_to_datetime_mixed(merged["official_date_detail"]))
    if require_completed:
        merged = merged.loc[
            (merged["status_code"] == 3)
            & (merged["winner_team_id"].notna())
            & (merged["is_regular_season"] == True)  # noqa: E712
            & (merged["official_date"].notna())
        ].copy()
    if require_completed and {"home_points", "away_points"}.issubset(merged.columns):
        merged = merged.loc[merged["home_points"].notna() & merged["away_points"].notna()].copy()
    merged["home_win"] = np.where(
        merged["winner_team_id"].notna(),
        (pd.to_numeric(merged["winner_team_id"], errors="coerce") == pd.to_numeric(merged["home_team_id"], errors="coerce")).astype(float),
        np.nan,
    )
    merged["away_win"] = np.where(merged["home_win"].notna(), 1.0 - merged["home_win"], np.nan)
    merged["month"] = merged["official_date"].dt.month
    merged["day_of_week"] = merged["official_date"].dt.dayofweek
    merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)
    merged["away_team_key"] = merged["league"].astype(str) + ":" + merged["away_team_id"].astype(str)
    merged["home_team_key"] = merged["league"].astype(str) + ":" + merged["home_team_id"].astype(str)
    merged = merged.sort_values(["official_date", "game_id"]).reset_index(drop=True)
    return merged


def _merge_latest_feature_rows(
    base: pd.DataFrame,
    history: pd.DataFrame,
    *,
    group_col: str,
    feature_cols: list[str],
    on_cols: list[str],
) -> pd.DataFrame:
    if history.empty or base.empty:
        return base

    output_parts: list[pd.DataFrame] = []
    history_frame = history.dropna(subset=[group_col, "official_date"]).copy()
    history_frame["official_date"] = _to_datetime_mixed(history_frame["official_date"])
    history_frame = history_frame.sort_values([group_col, "official_date", "game_id"]).reset_index(drop=True)

    for key, base_group in base.groupby(group_col, sort=False):
        history_group = history_frame.loc[history_frame[group_col] == key, ["official_date", *feature_cols]].copy()
        base_sorted = base_group.sort_values(["official_date", "game_id"]).reset_index(drop=True)
        if history_group.empty:
            for column in feature_cols:
                base_sorted[column] = np.nan
            output_parts.append(base_sorted)
            continue
        merged = pd.merge_asof(
            base_sorted,
            history_group.sort_values("official_date"),
            on="official_date",
            direction="backward",
            allow_exact_matches=False,
        )
        output_parts.append(merged)

    merged_base = pd.concat(output_parts, ignore_index=True)
    return merged_base.sort_values(on_cols).reset_index(drop=True)


def build_team_game_logs(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for side, opp in (("away", "home"), ("home", "away")):
            possessions = _estimate_possessions(
                game.get(f"{side}_field_goals_attempted"),
                game.get(f"{side}_free_throws_attempted"),
                game.get(f"{side}_rebounds_offensive"),
                game.get(f"{side}_turnovers"),
            )
            points_scored = pd.to_numeric(pd.Series([game.get(f"{side}_points")]), errors="coerce").iloc[0]
            points_allowed = pd.to_numeric(pd.Series([game.get(f"{opp}_points")]), errors="coerce").iloc[0]
            off_rating = _safe_divide((points_scored or 0) * 100.0, possessions)
            def_rating = _safe_divide((points_allowed or 0) * 100.0, possessions)
            net_rating = (off_rating - def_rating) if off_rating is not None and def_rating is not None else None
            rows.append(
                {
                    "game_id": game["game_id"],
                    "official_date": game["official_date"],
                    "league": game.get("league"),
                    "season_display": game.get("season_display"),
                    "team_side": side,
                    "team_id": game[f"{side}_team_id"],
                    "team_key": game[f"{side}_team_key"],
                    "team_name": game[f"{side}_team_name"],
                    "team_tricode": game.get(f"{side}_team_tricode"),
                    "opponent_team_id": game[f"{opp}_team_id"],
                    "is_home": 1 if side == "home" else 0,
                    "won": game.get(f"{side}_is_winner"),
                    "points_scored": points_scored,
                    "points_allowed": points_allowed,
                    "point_diff": (points_scored or 0) - (points_allowed or 0),
                    "assists": game.get(f"{side}_assists"),
                    "rebounds_total": game.get(f"{side}_rebounds_total"),
                    "rebounds_offensive": game.get(f"{side}_rebounds_offensive"),
                    "rebounds_defensive": game.get(f"{side}_rebounds_defensive"),
                    "turnovers": game.get(f"{side}_turnovers"),
                    "steals": game.get(f"{side}_steals"),
                    "blocks": game.get(f"{side}_blocks"),
                    "fouls_personal": game.get(f"{side}_fouls_personal"),
                    "field_goal_pct": game.get(f"{side}_field_goal_pct"),
                    "three_point_pct": game.get(f"{side}_three_point_pct"),
                    "free_throw_pct": game.get(f"{side}_free_throw_pct"),
                    "bench_points": game.get(f"{side}_bench_points"),
                    "fast_break_points": game.get(f"{side}_fast_break_points"),
                    "paint_points": game.get(f"{side}_paint_points"),
                    "second_chance_points": game.get(f"{side}_second_chance_points"),
                    "true_shooting_pct": game.get(f"{side}_true_shooting_pct"),
                    "effective_fg_pct": game.get(f"{side}_effective_fg_pct"),
                    "possessions_est": possessions,
                    "off_rating_est": off_rating,
                    "def_rating_est": def_rating,
                    "net_rating_est": net_rating,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["team_key", "official_date", "game_id"]).reset_index(drop=True)
    frame["official_date"] = _to_datetime_mixed(frame["official_date"])
    frame["days_rest"] = frame.groupby("team_key")["official_date"].diff().dt.days

    metrics = [
        "won",
        "points_scored",
        "points_allowed",
        "point_diff",
        "assists",
        "rebounds_total",
        "rebounds_offensive",
        "rebounds_defensive",
        "turnovers",
        "steals",
        "blocks",
        "fouls_personal",
        "field_goal_pct",
        "three_point_pct",
        "free_throw_pct",
        "bench_points",
        "fast_break_points",
        "paint_points",
        "second_chance_points",
        "true_shooting_pct",
        "effective_fg_pct",
        "possessions_est",
        "off_rating_est",
        "def_rating_est",
        "net_rating_est",
    ]
    windows = (3, 5, 10)
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        grouped = frame.groupby("team_key")[metric]
        for window in windows:
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    frame["games_played_prior"] = frame.groupby("team_key").cumcount()
    frame["win_pct_prior"] = frame.groupby("team_key")["won"].transform(lambda s: s.shift(1).expanding().mean())
    frame["same_site_win_pct_last_10"] = frame.groupby(["team_key", "is_home"])["won"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    return frame


def attach_pregame_team_features(games: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    if team_logs.empty or games.empty:
        return games

    safe_feature_columns = [
        column
        for column in team_logs.columns
        if column.endswith("_avg_last_3")
        or column.endswith("_avg_last_5")
        or column.endswith("_avg_last_10")
        or column in _SAFE_TEAM_LOG_BASE_COLUMNS
    ]
    general_columns = [column for column in safe_feature_columns if column != "same_site_win_pct_last_10"]

    merged = games.copy()
    side_specs = (
        ("away", "away_team_key", 0),
        ("home", "home_team_key", 1),
    )

    for side, team_col, is_home_value in side_specs:
        base = merged[["game_id", "official_date", team_col]].rename(columns={team_col: "team_key"})

        general = _merge_latest_feature_rows(
            base,
            team_logs[["team_key", "game_id", "official_date", *general_columns]],
            group_col="team_key",
            feature_cols=general_columns,
            on_cols=["official_date", "game_id"],
        ).rename(columns={column: f"{side}_team_{column}" for column in general_columns})
        general = general.rename(columns={"team_key": team_col})
        merged = merged.merge(general, on=["game_id", "official_date", team_col], how="left")

        same_site_history = team_logs.loc[team_logs["is_home"] == is_home_value, ["team_key", "game_id", "official_date", "same_site_win_pct_last_10"]]
        same_site = _merge_latest_feature_rows(
            base,
            same_site_history,
            group_col="team_key",
            feature_cols=["same_site_win_pct_last_10"],
            on_cols=["official_date", "game_id"],
        ).rename(columns={"same_site_win_pct_last_10": f"{side}_team_same_site_win_pct_last_10", "team_key": team_col})
        merged = merged.merge(same_site, on=["game_id", "official_date", team_col], how="left")

    return merged


def add_matchup_differentials(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    diff_count = 0
    differential_columns: dict[str, pd.Series] = {}
    prefix_pairs = (
        ("home_team_", "away_team_", "matchup_diff_"),
        ("home_rotation_", "away_rotation_", "rotation_diff_"),
        ("home_availability_", "away_availability_", "availability_diff_"),
    )
    for home_prefix, away_prefix, output_prefix in prefix_pairs:
        for column in list(frame.columns):
            if not column.startswith(home_prefix):
                continue
            suffix = column[len(home_prefix) :]
            away_column = f"{away_prefix}{suffix}"
            if away_column not in frame.columns:
                continue
            if frame[column].dtype.kind not in {"i", "u", "f", "b"} and frame[away_column].dtype.kind not in {"i", "u", "f", "b"}:
                continue
            target_column = f"{output_prefix}{suffix}"
            differential_columns[target_column] = (
                pd.to_numeric(frame[column], errors="coerce") - pd.to_numeric(frame[away_column], errors="coerce")
            )
            diff_count += 1
    if "home_team_days_rest" in frame.columns and "away_team_days_rest" in frame.columns:
        differential_columns["home_rest_advantage"] = (
            pd.to_numeric(frame["home_team_days_rest"], errors="coerce")
            - pd.to_numeric(frame["away_team_days_rest"], errors="coerce")
        )
    differential_columns["matchup_diff_count"] = pd.Series(diff_count, index=frame.index, dtype="int64")
    if differential_columns:
        frame = pd.concat([frame, pd.DataFrame(differential_columns, index=frame.index)], axis=1)
    return frame
