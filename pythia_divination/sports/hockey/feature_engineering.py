"""Feature engineering helpers for Hockey matchup modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_SAFE_TEAM_LOG_BASE_COLUMNS = {"games_played_prior", "win_pct_prior", "same_site_win_pct_last_5"}
_SAFE_GOALIE_LOG_BASE_COLUMNS = {"games_started_prior", "goalie_days_rest", "goalie_win_pct_prior"}



def _to_datetime_mixed(values: Any) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")



def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output



def prepare_games(
    schedule_df: pd.DataFrame,
    details_df: pd.DataFrame,
    *,
    season_start: int | None = None,
    season_end: int | None = None,
    require_completed: bool = True,
) -> pd.DataFrame:
    frame = schedule_df.copy()
    frame["season_start"] = pd.to_numeric(frame["season_start"], errors="coerce")
    if season_start is not None:
        frame = frame.loc[frame["season_start"] >= int(season_start)].copy()
    if season_end is not None:
        frame = frame.loc[frame["season_start"] <= int(season_end)].copy()
    frame["official_date"] = _to_datetime_mixed(frame.get("official_date", frame.get("game_date")))
    merged = frame.merge(details_df, on=["league", "game_id"], how="left", suffixes=("", "_detail"))
    if "official_date_detail" in merged.columns:
        merged["official_date"] = merged["official_date"].fillna(_to_datetime_mixed(merged["official_date_detail"]))
    merged = _safe_numeric(
        merged,
        [
            "game_type",
            "away_score",
            "home_score",
            "away_shots",
            "home_shots",
            "away_hits",
            "home_hits",
            "away_blocked_shots",
            "home_blocked_shots",
            "away_pim",
            "home_pim",
            "away_power_play_goals",
            "home_power_play_goals",
            "away_giveaways",
            "home_giveaways",
            "away_takeaways",
            "home_takeaways",
            "away_shooting_pct",
            "home_shooting_pct",
            "away_starter_goalie_save_pct",
            "home_starter_goalie_save_pct",
            "away_starter_goalie_saves",
            "home_starter_goalie_saves",
            "away_starter_goalie_shots_against",
            "home_starter_goalie_shots_against",
            "away_starter_goalie_goals_against",
            "home_starter_goalie_goals_against",
            "away_starter_goalie_toi_minutes",
            "home_starter_goalie_toi_minutes",
        ],
    )
    merged = merged.loc[merged["game_type"] == 2].copy()
    merged["home_win"] = np.where(
        merged["home_score"].notna() & merged["away_score"].notna(),
        (merged["home_score"] > merged["away_score"]).astype(float),
        np.nan,
    )
    merged["away_win"] = np.where(merged["home_win"].notna(), 1.0 - merged["home_win"], np.nan)
    merged["goal_diff_home"] = merged["home_score"] - merged["away_score"]
    merged["total_goals"] = merged["home_score"] + merged["away_score"]
    merged["is_weekend"] = merged["official_date"].dt.dayofweek.isin([5, 6]).astype(int)
    merged["month"] = merged["official_date"].dt.month
    merged["season_progress"] = merged["month"].map(lambda month: (month - 9) / 9.0 if pd.notna(month) else np.nan)
    merged["away_team_key"] = merged["away_team_key"].astype(str)
    merged["home_team_key"] = merged["home_team_key"].astype(str)
    if require_completed:
        merged = merged.loc[merged["home_win"].notna() & merged["official_date"].notna()].copy()
    return merged.sort_values(["official_date", "game_id"]).reset_index(drop=True)



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


_TEAM_BASE_METRICS = [
    "won",
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "shooting_pct",
    "save_pct",
    "hits",
    "blocked_shots",
    "pim",
    "power_play_goals",
    "giveaways",
    "takeaways",
]



def build_team_game_logs(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for side, opp in (("away", "home"), ("home", "away")):
            rows.append(
                {
                    "game_id": game["game_id"],
                    "official_date": game["official_date"],
                    "season_start": game["season_start"],
                    "season_display": game["season_display"],
                    "team_id": game.get(f"{side}_team_id"),
                    "team_key": game.get(f"{side}_team_key"),
                    "team_name": game.get(f"{side}_team_name"),
                    "opponent_team": game.get(f"{opp}_team_name"),
                    "is_home": 1 if side == "home" else 0,
                    "won": game.get(f"{side}_win"),
                    "goals_for": game.get(f"{side}_score"),
                    "goals_against": game.get(f"{opp}_score"),
                    "goal_diff": (game.get(f"{side}_score") or 0) - (game.get(f"{opp}_score") or 0),
                    "shots_for": game.get(f"{side}_shots"),
                    "shots_against": game.get(f"{opp}_shots"),
                    "shot_diff": (game.get(f"{side}_shots") or 0) - (game.get(f"{opp}_shots") or 0),
                    "shooting_pct": game.get(f"{side}_shooting_pct"),
                    "save_pct": game.get(f"{side}_starter_goalie_save_pct"),
                    "hits": game.get(f"{side}_hits"),
                    "blocked_shots": game.get(f"{side}_blocked_shots"),
                    "pim": game.get(f"{side}_pim"),
                    "power_play_goals": game.get(f"{side}_power_play_goals"),
                    "giveaways": game.get(f"{side}_giveaways"),
                    "takeaways": game.get(f"{side}_takeaways"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = _safe_numeric(frame, [column for column in _TEAM_BASE_METRICS if column in frame.columns])
    frame["official_date"] = _to_datetime_mixed(frame["official_date"])
    frame = frame.sort_values(["team_key", "official_date", "game_id"]).reset_index(drop=True)

    for metric in _TEAM_BASE_METRICS:
        grouped = frame.groupby("team_key")[metric]
        for window in (3, 5, 10):
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    frame["games_played_prior"] = frame.groupby("team_key").cumcount()
    frame["win_pct_prior"] = frame.groupby("team_key")["won"].transform(lambda series: series.shift(1).expanding().mean())
    frame["same_site_win_pct_last_5"] = frame.groupby(["team_key", "is_home"])["won"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    return frame


_GOALIE_BASE_METRICS = [
    "won",
    "save_pct",
    "goals_against",
    "shots_against",
    "saves",
    "toi_minutes",
]



def build_goalie_game_logs(player_boxscores: pd.DataFrame) -> pd.DataFrame:
    frame = player_boxscores.copy()
    if frame.empty:
        return frame
    frame = frame.loc[(frame["player_type"] == "goalie") & (pd.to_numeric(frame["starter"], errors="coerce") == 1)].copy()
    if frame.empty:
        return frame
    frame = _safe_numeric(frame, [column for column in _GOALIE_BASE_METRICS if column in frame.columns])
    frame["official_date"] = _to_datetime_mixed(frame["official_date"])
    frame["goalie_key"] = frame["player_id"].astype(str)
    frame = frame.sort_values(["goalie_key", "official_date", "game_id"]).reset_index(drop=True)

    for metric in _GOALIE_BASE_METRICS:
        grouped = frame.groupby("goalie_key")[metric]
        for window in (3, 5, 10):
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    frame["games_started_prior"] = frame.groupby("goalie_key").cumcount()
    frame["goalie_days_rest"] = frame.groupby("goalie_key")["official_date"].transform(lambda series: series.diff().dt.days)
    frame["goalie_win_pct_prior"] = frame.groupby("goalie_key")["won"].transform(lambda series: series.shift(1).expanding().mean())
    return frame



def attach_pregame_team_features(games: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    if games.empty or team_logs.empty:
        return games

    feature_columns = [
        column for column in team_logs.columns
        if column.endswith("_avg_last_3")
        or column.endswith("_avg_last_5")
        or column.endswith("_avg_last_10")
        or column in _SAFE_TEAM_LOG_BASE_COLUMNS
    ]
    general_columns = [column for column in feature_columns if column != "same_site_win_pct_last_5"]

    merged = games.copy()
    side_specs = (("away", "away_team_key", 0), ("home", "home_team_key", 1))
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

        same_site_history = team_logs.loc[team_logs["is_home"] == is_home_value, ["team_key", "game_id", "official_date", "same_site_win_pct_last_5"]]
        same_site = _merge_latest_feature_rows(
            base,
            same_site_history,
            group_col="team_key",
            feature_cols=["same_site_win_pct_last_5"],
            on_cols=["official_date", "game_id"],
        ).rename(columns={"same_site_win_pct_last_5": f"{side}_team_same_site_win_pct_last_5", "team_key": team_col})
        merged = merged.merge(same_site, on=["game_id", "official_date", team_col], how="left")

    return merged



def attach_pregame_goalie_features(games: pd.DataFrame, goalie_logs: pd.DataFrame) -> pd.DataFrame:
    if games.empty or goalie_logs.empty:
        return games

    feature_columns = [
        column for column in goalie_logs.columns
        if column.endswith("_avg_last_3")
        or column.endswith("_avg_last_5")
        or column.endswith("_avg_last_10")
        or column in _SAFE_GOALIE_LOG_BASE_COLUMNS
    ]
    history = goalie_logs[["team_key", "game_id", "official_date", *feature_columns]].copy()
    history = history.sort_values(["team_key", "official_date", "game_id"]).reset_index(drop=True)

    merged = games.copy()
    for side, team_col in (("away", "away_team_key"), ("home", "home_team_key")):
        base = merged[["game_id", "official_date", team_col]].rename(columns={team_col: "team_key"})
        latest = _merge_latest_feature_rows(
            base,
            history,
            group_col="team_key",
            feature_cols=feature_columns,
            on_cols=["official_date", "game_id"],
        ).rename(columns={column: f"{side}_goalie_{column}" for column in feature_columns})
        latest = latest.rename(columns={"team_key": team_col})
        merged = merged.merge(latest, on=["game_id", "official_date", team_col], how="left")

    return merged



def add_matchup_differentials(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    differential_columns: dict[str, pd.Series] = {}
    diff_count = 0
    for home_prefix, away_prefix, output_prefix in (
        ("home_team_", "away_team_", "matchup_diff_"),
        ("home_goalie_", "away_goalie_", "goalie_diff_"),
    ):
        for column in list(frame.columns):
            if not column.startswith(home_prefix):
                continue
            suffix = column[len(home_prefix):]
            away_column = f"{away_prefix}{suffix}"
            if away_column not in frame.columns:
                continue
            if frame[column].dtype.kind not in {"i", "u", "f", "b"} and frame[away_column].dtype.kind not in {"i", "u", "f", "b"}:
                continue
            target = f"{output_prefix}{suffix}"
            differential_columns[target] = pd.to_numeric(frame[column], errors="coerce") - pd.to_numeric(frame[away_column], errors="coerce")
            diff_count += 1
    differential_columns["matchup_diff_count"] = pd.Series(diff_count, index=frame.index, dtype="int64")
    if differential_columns:
        frame = pd.concat([frame, pd.DataFrame(differential_columns, index=frame.index)], axis=1)
    return frame
