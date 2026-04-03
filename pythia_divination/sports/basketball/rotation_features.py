"""Player-log and rotation feature helpers for basketball matchup modeling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(value: float | None, *, scale: float) -> float:
    if value is None or scale <= 0:
        return 0.0
    return float(max(0.0, min(100.0, (value / scale) * 100.0)))


def _shooting_score(row: Any) -> float:
    components = []
    for attr, weight in (
        ("field_goal_pct_avg_last_5", 0.45),
        ("three_point_pct_avg_last_5", 0.35),
        ("free_throw_pct_avg_last_5", 0.20),
    ):
        value = _safe_float(getattr(row, attr, np.nan))
        if value is not None:
            components.append((value, weight))
    if not components:
        return 0.0
    weighted = sum(value * weight for value, weight in components) / sum(weight for _, weight in components)
    return float(max(0.0, min(100.0, weighted * 100.0)))


def _player_radar_metrics_from_row(row: Any) -> list[dict[str, float]]:
    assists = _safe_float(getattr(row, "assists_avg_last_5", np.nan))
    rebounds = _safe_float(getattr(row, "rebounds_total_avg_last_5", np.nan))
    steals = _safe_float(getattr(row, "steals_avg_last_5", np.nan))
    blocks = _safe_float(getattr(row, "blocks_avg_last_5", np.nan))
    points = _safe_float(getattr(row, "points_recent", np.nan))
    minutes = _safe_float(getattr(row, "minutes_recent", np.nan))
    availability = _safe_float(getattr(row, "available_recent", np.nan))
    defense_value = (steals or 0.0) + 1.35 * (blocks or 0.0)
    return [
        {"label": "Scoring", "value": _clamp_score(points, scale=30.0)},
        {"label": "Playmaking", "value": _clamp_score(assists, scale=9.0)},
        {"label": "Rebounding", "value": _clamp_score(rebounds, scale=14.0)},
        {"label": "Defense", "value": _clamp_score(defense_value, scale=4.0)},
        {"label": "Shooting", "value": _shooting_score(row)},
        {"label": "Availability", "value": float(max(0.0, min(100.0, (availability or 0.0) * 100.0)))},
    ]


def build_player_game_logs(player_games: pd.DataFrame) -> pd.DataFrame:
    """Convert raw player boxscore rows into rolling player history."""
    frame = player_games.copy()
    if frame.empty:
        return frame

    frame["official_date"] = pd.to_datetime(frame["game_date_time_utc"], errors="coerce").dt.tz_localize(None)
    frame["player_key"] = frame["league"].astype(str) + ":" + frame["player_id"].astype(str)
    frame["team_key"] = frame["league"].astype(str) + ":" + frame["team_id"].astype(str)
    frame["starter"] = frame["starter"].fillna(False).astype(float)
    frame["played"] = frame["played"].fillna(False).astype(float)
    status_normalized = frame["status"].fillna("").astype(str).str.upper().str.strip()
    frame["available"] = np.where(status_normalized == "ACTIVE", 1.0, 0.0)
    frame["inactive"] = np.where(status_normalized == "INACTIVE", 1.0, 0.0)
    frame["active_dnp"] = np.where((frame["available"] == 1.0) & (frame["played"] == 0.0), 1.0, 0.0)
    frame = frame.sort_values(["player_key", "official_date", "game_id"]).reset_index(drop=True)
    frame["days_rest"] = frame.groupby("player_key")["official_date"].diff().dt.days

    metrics = [
        "minutes",
        "points",
        "assists",
        "rebounds_total",
        "rebounds_offensive",
        "rebounds_defensive",
        "steals",
        "blocks",
        "turnovers",
        "field_goal_pct",
        "three_point_pct",
        "free_throw_pct",
        "plus_minus_points",
        "starter",
        "played",
        "available",
        "inactive",
        "active_dnp",
    ]
    windows = (3, 5, 10)
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        grouped = frame.groupby("player_key")[metric]
        for window in windows:
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    frame["games_played_prior"] = frame.groupby("player_key").cumcount()
    return frame


def _overlap_count(current_ids: Iterable[int], previous_ids: Iterable[int]) -> int:
    current = {int(player_id) for player_id in current_ids if pd.notna(player_id)}
    previous = {int(player_id) for player_id in previous_ids if pd.notna(player_id)}
    return len(current & previous)


def build_rotation_game_logs(player_games: pd.DataFrame) -> pd.DataFrame:
    """Summarize one team-game rotation into lineup continuity and usage features."""
    frame = player_games.copy()
    if frame.empty:
        return frame

    frame["official_date"] = pd.to_datetime(frame["game_date_time_utc"], errors="coerce").dt.tz_localize(None)
    frame["team_key"] = frame["league"].astype(str) + ":" + frame["team_id"].astype(str)
    frame["starter"] = frame["starter"].fillna(False).astype(bool)
    frame["played"] = frame["played"].fillna(False).astype(bool)
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce")
    frame["points"] = pd.to_numeric(frame["points"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for (league, team_key, team_id, game_id), group in frame.groupby(
        ["league", "team_key", "team_id", "game_id"], sort=False
    ):
        group = group.sort_values(["minutes", "points"], ascending=[False, False]).reset_index(drop=True)
        played = group.loc[group["played"] == True].copy()  # noqa: E712
        starters = played.loc[played["starter"] == True].copy()  # noqa: E712
        bench = played.loc[played["starter"] != True].copy()  # noqa: E712

        total_minutes = played["minutes"].sum(min_count=1)
        total_points = played["points"].sum(min_count=1)
        top5 = played.head(5)
        top8 = played.head(8)
        top3_points = played.sort_values("points", ascending=False).head(3)

        rows.append(
            {
                "league": league,
                "team_key": team_key,
                "team_id": team_id,
                "game_id": game_id,
                "official_date": group["official_date"].iloc[0],
                "players_used": int(len(played)),
                "starters_used": int(len(starters)),
                "bench_players_used": int(len(bench)),
                "starter_minutes_total": starters["minutes"].sum(min_count=1),
                "bench_minutes_total": bench["minutes"].sum(min_count=1),
                "starter_points_total": starters["points"].sum(min_count=1),
                "bench_points_total": bench["points"].sum(min_count=1),
                "top_5_minutes_share": (top5["minutes"].sum(min_count=1) / total_minutes) if pd.notna(total_minutes) and total_minutes else np.nan,
                "top_8_minutes_share": (top8["minutes"].sum(min_count=1) / total_minutes) if pd.notna(total_minutes) and total_minutes else np.nan,
                "top_3_points_share": (top3_points["points"].sum(min_count=1) / total_points) if pd.notna(total_points) and total_points else np.nan,
                "minutes_from_starters_share": (starters["minutes"].sum(min_count=1) / total_minutes) if pd.notna(total_minutes) and total_minutes else np.nan,
                "points_from_bench_share": (bench["points"].sum(min_count=1) / total_points) if pd.notna(total_points) and total_points else np.nan,
                "rotation_player_ids": played["player_id"].dropna().astype(int).tolist(),
                "starter_player_ids": starters["player_id"].dropna().astype(int).tolist(),
            }
        )

    rotation = pd.DataFrame(rows)
    if rotation.empty:
        return rotation

    rotation = rotation.sort_values(["team_key", "official_date", "game_id"]).reset_index(drop=True)
    rotation["days_rest"] = rotation.groupby("team_key")["official_date"].diff().dt.days
    rotation["prev_rotation_overlap"] = (
        rotation.groupby("team_key")["rotation_player_ids"]
        .transform(lambda series: [np.nan] + [_overlap_count(current, previous) for previous, current in zip(series[:-1], series[1:])])
    )
    rotation["prev_starter_overlap"] = (
        rotation.groupby("team_key")["starter_player_ids"]
        .transform(lambda series: [np.nan] + [_overlap_count(current, previous) for previous, current in zip(series[:-1], series[1:])])
    )

    metrics = [
        "players_used",
        "starters_used",
        "bench_players_used",
        "starter_minutes_total",
        "bench_minutes_total",
        "starter_points_total",
        "bench_points_total",
        "top_5_minutes_share",
        "top_8_minutes_share",
        "top_3_points_share",
        "minutes_from_starters_share",
        "points_from_bench_share",
        "prev_rotation_overlap",
        "prev_starter_overlap",
    ]
    windows = (3, 5, 10)
    for metric in metrics:
        rotation[metric] = pd.to_numeric(rotation[metric], errors="coerce")
        grouped = rotation.groupby("team_key")[metric]
        for window in windows:
            rotation[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    rotation["games_played_prior"] = rotation.groupby("team_key").cumcount()
    return rotation.drop(columns=["rotation_player_ids", "starter_player_ids"])


def _empty_expected_rotation_summary() -> dict[str, float | int | None]:
    return {
        "expected_available_players": 0,
        "expected_rotation_players": 0,
        "expected_starters": 0,
        "likely_inactive_core_players": 0,
        "likely_absent_rotation_players": 0,
        "expected_rotation_minutes_total": np.nan,
        "expected_rotation_points_total": np.nan,
        "expected_starter_minutes_total": np.nan,
        "expected_top_5_minutes_share": np.nan,
        "expected_top_8_minutes_share": np.nan,
        "expected_top_3_points_share": np.nan,
        "likely_inactive_core_minutes": np.nan,
        "likely_inactive_core_points": np.nan,
        "likely_absent_rotation_minutes": np.nan,
        "likely_absent_rotation_points": np.nan,
        "expected_lineup_continuity_last_game": np.nan,
        "expected_starter_continuity_last_game": np.nan,
        "expected_rotation_confidence": np.nan,
        "availability_rating": np.nan,
        "core_availability_rating": np.nan,
    }


def _annotate_expected_player_state(latest_rows: pd.DataFrame) -> pd.DataFrame:
    if latest_rows.empty:
        return latest_rows.copy()

    frame = latest_rows.copy()
    numeric_columns = [
        "minutes",
        "points",
        "starter",
        "played",
        "available",
        "inactive",
        "active_dnp",
        "minutes_avg_last_5",
        "minutes_avg_last_10",
        "points_avg_last_5",
        "points_avg_last_10",
        "starter_avg_last_5",
        "starter_avg_last_10",
        "played_avg_last_3",
        "played_avg_last_5",
        "available_avg_last_3",
        "available_avg_last_5",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["minutes_recent"] = frame["minutes_avg_last_5"].fillna(frame["minutes_avg_last_10"]).fillna(frame["minutes"]).fillna(0.0)
    frame["points_recent"] = frame["points_avg_last_5"].fillna(frame["points_avg_last_10"]).fillna(frame["points"]).fillna(0.0)
    frame["starter_recent"] = frame["starter_avg_last_5"].fillna(frame["starter_avg_last_10"]).fillna(frame["starter"]).fillna(0.0)
    frame["available_recent"] = frame["available_avg_last_3"].fillna(frame["available_avg_last_5"]).fillna(frame["available"]).fillna(0.0)
    frame["played_recent"] = frame["played_avg_last_3"].fillna(frame["played_avg_last_5"]).fillna(frame["played"]).fillna(0.0)

    frame["is_core_player"] = frame["minutes_avg_last_10"].fillna(frame["minutes_recent"]) >= 20.0
    frame["is_rotation_player"] = frame["minutes_recent"] >= 8.0
    frame["expected_available"] = (frame["available"].fillna(0.0) >= 1.0) | (frame["available_recent"] >= 0.67)
    frame["expected_rotation"] = frame["expected_available"] & frame["is_rotation_player"]
    frame["expected_starter"] = frame["expected_available"] & (
        (frame["starter_recent"] >= 0.4) | ((frame["starter"].fillna(0.0) >= 1.0) & (frame["minutes_recent"] >= 18.0))
    )
    frame["likely_inactive_core"] = (~frame["expected_available"]) & frame["is_core_player"]
    frame["likely_absent_rotation"] = (~frame["expected_available"]) & frame["is_rotation_player"]
    return frame


def _summarize_expected_rotation_state(latest_rows: pd.DataFrame) -> dict[str, float | int | None]:
    if latest_rows.empty:
        return _empty_expected_rotation_summary()

    frame = _annotate_expected_player_state(latest_rows)

    expected_rotation = frame.loc[frame["expected_rotation"]].sort_values(["minutes_recent", "points_recent"], ascending=False)
    expected_starters = frame.loc[frame["expected_starter"]]
    core_absences = frame.loc[frame["likely_inactive_core"]]
    rotation_absences = frame.loc[frame["likely_absent_rotation"]]

    expected_minutes_total = expected_rotation["minutes_recent"].sum(min_count=1)
    expected_points_total = expected_rotation["points_recent"].sum(min_count=1)
    top5 = expected_rotation.head(5)
    top8 = expected_rotation.head(8)
    top3_points = expected_rotation.sort_values("points_recent", ascending=False).head(3)

    lineup_continuity = pd.to_numeric(
        frame.loc[frame["expected_rotation"], "played"].fillna(0.0),
        errors="coerce",
    )
    starter_continuity = pd.to_numeric(
        frame.loc[frame["expected_starter"], "starter"].fillna(0.0),
        errors="coerce",
    )
    confidence_candidates = pd.concat(
        [
            frame.loc[frame["expected_rotation"], "available_recent"],
            frame.loc[frame["expected_rotation"], "played_recent"],
        ],
        axis=0,
    )

    return {
        "expected_available_players": int(frame["expected_available"].sum()),
        "expected_rotation_players": int(frame["expected_rotation"].sum()),
        "expected_starters": int(frame["expected_starter"].sum()),
        "likely_inactive_core_players": int(frame["likely_inactive_core"].sum()),
        "likely_absent_rotation_players": int(frame["likely_absent_rotation"].sum()),
        "expected_rotation_minutes_total": float(expected_minutes_total) if pd.notna(expected_minutes_total) else np.nan,
        "expected_rotation_points_total": float(expected_points_total) if pd.notna(expected_points_total) else np.nan,
        "expected_starter_minutes_total": float(expected_starters["minutes_recent"].sum(min_count=1))
        if not expected_starters.empty
        else np.nan,
        "expected_top_5_minutes_share": float(top5["minutes_recent"].sum(min_count=1) / expected_minutes_total)
        if pd.notna(expected_minutes_total) and expected_minutes_total
        else np.nan,
        "expected_top_8_minutes_share": float(top8["minutes_recent"].sum(min_count=1) / expected_minutes_total)
        if pd.notna(expected_minutes_total) and expected_minutes_total
        else np.nan,
        "expected_top_3_points_share": float(top3_points["points_recent"].sum(min_count=1) / expected_points_total)
        if pd.notna(expected_points_total) and expected_points_total
        else np.nan,
        "likely_inactive_core_minutes": float(core_absences["minutes_avg_last_10"].sum(min_count=1)) if not core_absences.empty else np.nan,
        "likely_inactive_core_points": float(core_absences["points_avg_last_10"].sum(min_count=1)) if not core_absences.empty else np.nan,
        "likely_absent_rotation_minutes": float(rotation_absences["minutes_recent"].sum(min_count=1)) if not rotation_absences.empty else np.nan,
        "likely_absent_rotation_points": float(rotation_absences["points_recent"].sum(min_count=1)) if not rotation_absences.empty else np.nan,
        "expected_lineup_continuity_last_game": float(lineup_continuity.mean()) if not lineup_continuity.empty else np.nan,
        "expected_starter_continuity_last_game": float(starter_continuity.mean()) if not starter_continuity.empty else np.nan,
        "expected_rotation_confidence": float(confidence_candidates.mean()) if not confidence_candidates.empty else np.nan,
        "availability_rating": float(frame["available_recent"].mean()) if not frame.empty else np.nan,
        "core_availability_rating": float(frame.loc[frame["is_core_player"], "available_recent"].mean())
        if frame["is_core_player"].any()
        else np.nan,
    }


def build_projected_rotation_map(
    games: pd.DataFrame,
    player_logs: pd.DataFrame,
    *,
    top_n: int = 9,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Build projected top-rotation players for each upcoming basketball game side."""
    if games.empty or player_logs.empty:
        return {}

    rotation_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    history = player_logs.sort_values(["team_key", "official_date", "game_id", "player_key"]).reset_index(drop=True)

    for game in games.to_dict(orient="records"):
        game_id = str(game["game_id"])
        game_date = pd.to_datetime(game["official_date"], errors="coerce")
        if pd.isna(game_date):
            continue
        rotation_map.setdefault(game_id, {})
        for side, team_col in (("away", "away_team_key"), ("home", "home_team_key")):
            team_key = str(game.get(team_col) or "")
            team_history = history.loc[history["team_key"] == team_key].copy()
            if team_history.empty:
                rotation_map[game_id][side] = []
                continue

            team_history = team_history.loc[pd.to_datetime(team_history["official_date"], errors="coerce") < game_date].copy()
            if team_history.empty:
                rotation_map[game_id][side] = []
                continue

            latest_rows = (
                team_history.sort_values(["official_date", "game_id", "player_key"])
                .groupby("player_key", as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )
            annotated = _annotate_expected_player_state(latest_rows)
            projected = (
                annotated.loc[annotated["expected_rotation"]]
                .sort_values(["starter_recent", "minutes_recent", "points_recent"], ascending=[False, False, False])
                .head(top_n)
            )

            entries: list[dict[str, Any]] = []
            for index, row in enumerate(projected.itertuples(index=False), start=1):
                assists_recent = _safe_float(getattr(row, "assists_avg_last_5", np.nan))
                rebounds_recent = _safe_float(getattr(row, "rebounds_total_avg_last_5", np.nan))
                field_goal_pct_recent = _safe_float(getattr(row, "field_goal_pct_avg_last_5", np.nan))
                entries.append(
                    {
                        "playerId": int(row.player_id) if pd.notna(row.player_id) else None,
                        "playerName": str(row.player_name),
                        "lineupSlot": index,
                        "position": str(row.position) if pd.notna(row.position) else None,
                        "performanceSummary": (
                            f"{float(row.minutes_recent):.1f} min | "
                            f"{float(row.points_recent):.1f} pts | "
                            f"{float(row.played_recent) * 100:.0f}% recent availability"
                        ),
                        "profile": {
                            "imageUrl": (
                                f"https://cdn.wnba.com/headshots/wnba/latest/1040x760/{int(row.player_id)}.png"
                                if str(row.league) == "wnba" and pd.notna(row.player_id)
                                else (
                                    f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(row.player_id)}.png"
                                    if pd.notna(row.player_id)
                                    else None
                                )
                            ),
                            "subtitle": (
                                "Projected starter"
                                if float(row.starter_recent) >= 0.5
                                else "Projected rotation"
                            ),
                            "stats": [
                                {"label": "Min L5", "value": f"{float(row.minutes_avg_last_5):.1f}"}
                                if pd.notna(row.minutes_avg_last_5)
                                else {"label": "Min", "value": f"{float(row.minutes_recent):.1f}"},
                                {"label": "Pts L5", "value": f"{float(row.points_avg_last_5):.1f}"}
                                if pd.notna(row.points_avg_last_5)
                                else {"label": "Pts", "value": f"{float(row.points_recent):.1f}"},
                                {"label": "FG% L5", "value": f"{field_goal_pct_recent * 100:.1f}%"}
                                if field_goal_pct_recent is not None
                                else {"label": "Avail", "value": f"{float(row.available_recent) * 100:.0f}%"},
                                {"label": "Ast L5", "value": f"{assists_recent:.1f}"}
                                if assists_recent is not None
                                else {"label": "Start %", "value": f"{float(row.starter_recent) * 100:.0f}%"},
                                {"label": "Reb L5", "value": f"{rebounds_recent:.1f}"}
                                if rebounds_recent is not None
                                else {"label": "Avail+", "value": f"{float(row.available_recent) * 100:.0f}%"},
                            ],
                        },
                        "radarMetrics": _player_radar_metrics_from_row(row),
                    }
                )
            rotation_map[game_id][side] = entries

    return rotation_map


def build_expected_rotation_game_logs(games: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    """Estimate pregame availability and expected rotation state for each team-game."""
    if games.empty or player_logs.empty:
        return pd.DataFrame()

    team_games: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for side in ("away", "home"):
            team_games.append(
                {
                    "league": game["league"],
                    "team_key": game[f"{side}_team_key"],
                    "team_id": game[f"{side}_team_id"],
                    "game_id": game["game_id"],
                    "official_date": game["official_date"],
                }
            )

    base = pd.DataFrame(team_games).sort_values(["team_key", "official_date", "game_id"]).reset_index(drop=True)
    history = player_logs.sort_values(["team_key", "official_date", "game_id", "player_key"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for team_key, base_group in base.groupby("team_key", sort=False):
        team_history = history.loc[history["team_key"] == team_key].copy()
        if team_history.empty:
            for record in base_group.to_dict(orient="records"):
                rows.append({**record, **_summarize_expected_rotation_state(pd.DataFrame())})
            continue

        history_records = team_history.to_dict(orient="records")
        latest_by_player: dict[str, dict[str, Any]] = {}
        pointer = 0

        for record in base_group.sort_values(["official_date", "game_id"]).to_dict(orient="records"):
            record_date = pd.to_datetime(record["official_date"], errors="coerce")
            while pointer < len(history_records):
                candidate = history_records[pointer]
                candidate_date = pd.to_datetime(candidate["official_date"], errors="coerce")
                if pd.isna(candidate_date) or candidate_date >= record_date:
                    break
                latest_by_player[str(candidate["player_key"])] = candidate
                pointer += 1

            latest_frame = pd.DataFrame(list(latest_by_player.values()))
            rows.append({**record, **_summarize_expected_rotation_state(latest_frame)})

    return pd.DataFrame(rows)


def attach_pregame_rotation_features(games: pd.DataFrame, rotation_logs: pd.DataFrame) -> pd.DataFrame:
    """Attach prior team rotation summaries to game rows."""
    if rotation_logs.empty or games.empty:
        return games

    safe_feature_columns = [
        column
        for column in rotation_logs.columns
        if column.endswith("_avg_last_3")
        or column.endswith("_avg_last_5")
        or column.endswith("_avg_last_10")
        or column in {"days_rest", "games_played_prior"}
    ]

    merged = games.copy()
    for side, team_col in (("away", "away_team_key"), ("home", "home_team_key")):
        base = merged[["game_id", "official_date", team_col]].rename(columns={team_col: "team_key"})
        output_parts: list[pd.DataFrame] = []
        history = rotation_logs[["team_key", "game_id", "official_date", *safe_feature_columns]].copy()
        history["official_date"] = pd.to_datetime(history["official_date"], errors="coerce")
        history = history.sort_values(["team_key", "official_date", "game_id"]).reset_index(drop=True)

        for team_key, base_group in base.groupby("team_key", sort=False):
            history_group = history.loc[history["team_key"] == team_key, ["official_date", *safe_feature_columns]].copy()
            base_sorted = base_group.sort_values(["official_date", "game_id"]).reset_index(drop=True)
            if history_group.empty:
                for column in safe_feature_columns:
                    base_sorted[column] = np.nan
                output_parts.append(base_sorted)
                continue
            output_parts.append(
                pd.merge_asof(
                    base_sorted,
                    history_group.sort_values("official_date"),
                    on="official_date",
                    direction="backward",
                    allow_exact_matches=False,
                )
            )

        feature_frame = pd.concat(output_parts, ignore_index=True)
        feature_frame = feature_frame.rename(
            columns={
                "team_key": team_col,
                **{column: f"{side}_rotation_{column}" for column in safe_feature_columns},
            }
        )
        merged = merged.merge(feature_frame, on=["game_id", "official_date", team_col], how="left")

    return merged


def attach_expected_rotation_features(games: pd.DataFrame, expected_logs: pd.DataFrame) -> pd.DataFrame:
    """Attach precomputed expected availability / rotation summaries to game rows."""
    if expected_logs.empty or games.empty:
        return games

    merged = games.copy()
    feature_columns = [
        column
        for column in expected_logs.columns
        if column not in {"league", "team_key", "team_id", "game_id", "official_date"}
    ]

    for side, team_col in (("away", "away_team_key"), ("home", "home_team_key")):
        feature_frame = expected_logs[["game_id", "official_date", "team_key", *feature_columns]].copy().rename(
            columns={
                "team_key": team_col,
                **{column: f"{side}_availability_{column}" for column in feature_columns},
            }
        )
        merged = merged.merge(feature_frame, on=["game_id", "official_date", team_col], how="left")

    return merged
