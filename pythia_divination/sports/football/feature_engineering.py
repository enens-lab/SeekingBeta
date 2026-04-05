"""Feature engineering helpers for Football matchup modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_SAFE_ROOF_VALUES = {"dome", "closed", "indoor"}
_GRASS_SURFACE_HINTS = ("grass", "bermuda")
_TURF_SURFACE_HINTS = ("turf", "astro", "artificial")


def _to_datetime(values: Any) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _safe_bool_to_int(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(int)


def prepare_games(
    games_df: pd.DataFrame,
    *,
    season_start: int | None = None,
    season_end: int | None = None,
    require_completed: bool = True,
) -> pd.DataFrame:
    frame = games_df.copy()
    if frame.empty:
        return frame

    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    if season_start is not None:
        frame = frame.loc[frame["season"] >= int(season_start)].copy()
    if season_end is not None:
        frame = frame.loc[frame["season"] <= int(season_end)].copy()

    frame = frame.loc[frame["game_type"] == "REG"].copy()
    frame["official_date"] = _to_datetime(frame["gameday"])
    frame = _safe_numeric(
        frame,
        [
            "away_score",
            "home_score",
            "away_rest",
            "home_rest",
            "away_moneyline",
            "home_moneyline",
            "spread_line",
            "total_line",
            "temp",
            "wind",
            "result",
            "total",
        ],
    )
    frame["div_game"] = _safe_bool_to_int(frame["div_game"])
    frame["overtime"] = _safe_bool_to_int(frame["overtime"])
    frame["is_neutral_site"] = frame["location"].fillna("").astype(str).str.lower().eq("neutral").astype(int)
    frame["is_weekend"] = frame["weekday"].fillna("").isin(["Saturday", "Sunday"]).astype(int)
    roof = frame["roof"].fillna("").astype(str).str.lower()
    surface = frame["surface"].fillna("").astype(str).str.lower()
    frame["roof_is_closed"] = roof.isin(_SAFE_ROOF_VALUES).astype(int)
    frame["surface_is_grass"] = surface.map(lambda value: int(any(hint in value for hint in _GRASS_SURFACE_HINTS)))
    frame["surface_is_turf"] = surface.map(lambda value: int(any(hint in value for hint in _TURF_SURFACE_HINTS)))
    frame["weather_temp"] = frame["temp"]
    frame["weather_wind"] = frame["wind"]
    frame["home_is_favorite"] = np.where(
        frame["home_moneyline"].notna() & frame["away_moneyline"].notna(),
        (frame["home_moneyline"] < frame["away_moneyline"]).astype(int),
        np.nan,
    )
    frame["home_win"] = np.where(
        frame["home_score"].notna() & frame["away_score"].notna(),
        (frame["home_score"] > frame["away_score"]).astype(float),
        np.nan,
    )
    frame["away_win"] = np.where(frame["home_win"].notna(), 1.0 - frame["home_win"], np.nan)
    frame["point_margin_home"] = frame["home_score"] - frame["away_score"]
    frame["point_total"] = frame["home_score"] + frame["away_score"]
    frame["season_progress"] = frame["week"] / 18.0
    frame["away_team_key"] = frame["away_team"].astype(str)
    frame["home_team_key"] = frame["home_team"].astype(str)
    frame["season_display"] = frame["season"].astype("Int64").astype(str)
    if require_completed:
        frame = frame.loc[frame["home_win"].notna() & frame["official_date"].notna()].copy()
    return frame.sort_values(["official_date", "season", "week", "game_id"]).reset_index(drop=True)


_TEAM_BASE_METRICS = [
    "won",
    "points_scored",
    "points_allowed",
    "point_diff",
    "days_rest",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_epa",
    "passing_cpoe",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "receiving_yards",
    "receiving_tds",
    "receiving_epa",
    "def_sacks",
    "def_interceptions",
    "def_pass_defended",
    "def_tackles_for_loss",
    "penalties",
    "penalty_yards",
    "fg_pct",
    "total_yards",
    "offense_epa_total",
    "explosiveness_index",
]


def build_team_game_logs(games: pd.DataFrame, team_week_stats: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    stats = team_week_stats.copy()
    if stats.empty:
        stats = pd.DataFrame(columns=["season", "week", "team", "game_id"])
    stats = stats.loc[stats.get("season_type", "REG") == "REG"].copy()
    numeric_candidate_columns = [
        "season",
        "week",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "sacks_suffered",
        "passing_epa",
        "passing_cpoe",
        "rushing_yards",
        "rushing_tds",
        "rushing_epa",
        "receiving_yards",
        "receiving_tds",
        "receiving_epa",
        "def_sacks",
        "def_interceptions",
        "def_pass_defended",
        "def_tackles_for_loss",
        "penalties",
        "penalty_yards",
        "fg_pct",
    ]
    stats = _safe_numeric(stats, [column for column in numeric_candidate_columns if column in stats.columns])
    stats["key"] = (
        stats["season"].astype("Int64").astype(str)
        + ":" + stats["week"].astype("Int64").astype(str)
        + ":" + stats["team"].astype(str)
        + ":" + stats["game_id"].astype(str)
    )
    stats_map = stats.set_index("key").to_dict(orient="index") if not stats.empty else {}

    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for side, opp in (("away", "home"), ("home", "away")):
            key = f"{int(game['season'])}:{int(game['week'])}:{game[f'{side}_team']}:{game['game_id']}"
            stat_row = stats_map.get(key, {})
            passing_yards = pd.to_numeric(pd.Series([stat_row.get("passing_yards")]), errors="coerce").iloc[0]
            rushing_yards = pd.to_numeric(pd.Series([stat_row.get("rushing_yards")]), errors="coerce").iloc[0]
            passing_epa = pd.to_numeric(pd.Series([stat_row.get("passing_epa")]), errors="coerce").iloc[0]
            rushing_epa = pd.to_numeric(pd.Series([stat_row.get("rushing_epa")]), errors="coerce").iloc[0]
            rows.append(
                {
                    "game_id": game["game_id"],
                    "official_date": game["official_date"],
                    "season": game["season"],
                    "season_display": game["season_display"],
                    "week": game["week"],
                    "team": game[f"{side}_team"],
                    "team_key": game[f"{side}_team_key"],
                    "opponent_team": game[f"{opp}_team"],
                    "is_home": 1 if side == "home" else 0,
                    "won": game[f"{side}_win"],
                    "points_scored": game[f"{side}_score"],
                    "points_allowed": game[f"{opp}_score"],
                    "point_diff": game[f"{side}_score"] - game[f"{opp}_score"],
                    "days_rest": game.get(f"{side}_rest"),
                    "qb_id": game.get(f"{side}_qb_id"),
                    "qb_name": game.get(f"{side}_qb_name"),
                    "passing_yards": passing_yards,
                    "passing_tds": stat_row.get("passing_tds"),
                    "passing_interceptions": stat_row.get("passing_interceptions"),
                    "sacks_suffered": stat_row.get("sacks_suffered"),
                    "passing_epa": passing_epa,
                    "passing_cpoe": stat_row.get("passing_cpoe"),
                    "rushing_yards": rushing_yards,
                    "rushing_tds": stat_row.get("rushing_tds"),
                    "rushing_epa": rushing_epa,
                    "receiving_yards": stat_row.get("receiving_yards"),
                    "receiving_tds": stat_row.get("receiving_tds"),
                    "receiving_epa": stat_row.get("receiving_epa"),
                    "def_sacks": stat_row.get("def_sacks"),
                    "def_interceptions": stat_row.get("def_interceptions"),
                    "def_pass_defended": stat_row.get("def_pass_defended"),
                    "def_tackles_for_loss": stat_row.get("def_tackles_for_loss"),
                    "penalties": stat_row.get("penalties"),
                    "penalty_yards": stat_row.get("penalty_yards"),
                    "fg_pct": stat_row.get("fg_pct"),
                    "total_yards": (passing_yards if pd.notna(passing_yards) else 0.0) + (rushing_yards if pd.notna(rushing_yards) else 0.0),
                    "offense_epa_total": (passing_epa if pd.notna(passing_epa) else 0.0) + (rushing_epa if pd.notna(rushing_epa) else 0.0),
                    "explosiveness_index": (
                        ((passing_yards if pd.notna(passing_yards) else 0.0) + (rushing_yards if pd.notna(rushing_yards) else 0.0))
                        / max(1.0, (pd.to_numeric(pd.Series([stat_row.get('attempts')]), errors='coerce').iloc[0] if pd.notna(pd.to_numeric(pd.Series([stat_row.get('attempts')]), errors='coerce').iloc[0]) else 0.0)
                          + (pd.to_numeric(pd.Series([stat_row.get('carries')]), errors='coerce').iloc[0] if pd.notna(pd.to_numeric(pd.Series([stat_row.get('carries')]), errors='coerce').iloc[0]) else 0.0))
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = _safe_numeric(frame, [column for column in _TEAM_BASE_METRICS if column in frame.columns])
    frame["official_date"] = _to_datetime(frame["official_date"])
    frame = frame.sort_values(["team_key", "official_date", "game_id"]).reset_index(drop=True)

    windows = (3, 5, 10)
    for metric in _TEAM_BASE_METRICS:
        grouped = frame.groupby("team_key")[metric]
        for window in windows:
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    frame["games_played_prior"] = frame.groupby("team_key").cumcount()
    frame["win_pct_prior"] = frame.groupby("team_key")["won"].transform(lambda s: s.shift(1).expanding().mean())
    frame["same_site_win_pct_last_5"] = frame.groupby(["team_key", "is_home"])["won"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    return frame


_QB_BASE_METRICS = [
    "games",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_epa",
    "passing_cpoe",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "fantasy_points",
    "fantasy_points_ppr",
]


def build_qb_week_logs(player_week_stats: pd.DataFrame) -> pd.DataFrame:
    frame = player_week_stats.copy()
    if frame.empty:
        return frame

    frame = frame.loc[frame["season_type"] == "REG"].copy()
    frame = frame.loc[
        frame["position_group"].fillna("").eq("QB") | frame["position"].fillna("").eq("QB")
    ].copy()
    if frame.empty:
        return frame

    numeric_columns = ["season", "week", *[column for column in _QB_BASE_METRICS if column in frame.columns]]
    frame = _safe_numeric(frame, numeric_columns)
    frame["official_date"] = pd.NaT
    frame["player_key"] = frame["player_id"].astype(str)
    frame = frame.sort_values(["player_key", "season", "week", "game_id"]).reset_index(drop=True)

    windows = (3, 5, 10)
    for metric in [column for column in _QB_BASE_METRICS if column in frame.columns]:
        grouped = frame.groupby("player_key")[metric]
        for window in windows:
            frame[f"{metric}_avg_last_{window}"] = grouped.transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    frame["games_played_prior"] = frame.groupby("player_key").cumcount()
    return frame


_POSITION_BUCKETS = {
    "QB": "qb",
    "RB": "skill",
    "FB": "skill",
    "WR": "skill",
    "TE": "skill",
    "C": "offensive_line",
    "G": "offensive_line",
    "T": "offensive_line",
    "OL": "offensive_line",
    "DL": "front_seven",
    "DE": "front_seven",
    "DT": "front_seven",
    "NT": "front_seven",
    "LB": "front_seven",
    "EDGE": "front_seven",
    "CB": "secondary",
    "DB": "secondary",
    "FS": "secondary",
    "SS": "secondary",
    "S": "secondary",
    "K": "specialists",
    "P": "specialists",
    "LS": "specialists",
}
_ACTIVE_STATUS_HINTS = {"ACT", "A", "ACTIVE", "PROB", "QST"}
_INACTIVE_STATUS_HINTS = {"IR", "NFI", "PUP", "DNR", "SUS", "RET", "CUT", "EXE", "INA", "INACTIVE"}
_ACTIVE_STATUSES = {"ACT"}
_INACTIVE_STATUSES = {"INA"}
_RESERVE_STATUSES = {"RES", "PUP", "RSR", "RSN", "NWT"}
_PRACTICE_SQUAD_STATUSES = {"DEV"}
_SUSPENDED_STATUSES = {"SUS"}
_REMOVED_STATUSES = {"CUT", "RET", "TRD", "TRC", "TRT", "UFA", "RFA", "EXE"}


def _position_bucket(value: Any) -> str:
    return _POSITION_BUCKETS.get(str(value or "").strip().upper(), "other")


def _is_available_row(row: pd.Series) -> int:
    values = [str(row.get("status") or "").upper(), str(row.get("status_description_abbr") or "").upper()]
    if any(token in value for value in values for token in _INACTIVE_STATUS_HINTS):
        return 0
    if any(token in value for value in values for token in _ACTIVE_STATUS_HINTS):
        return 1
    status = str(row.get("status") or "").upper().strip()
    return 0 if status in {"DEV", "RESERVE"} else 1


def _status_bucket(value: Any) -> str:
    status = str(value or "").strip().upper()
    if status in _ACTIVE_STATUSES:
        return "active"
    if status in _INACTIVE_STATUSES:
        return "inactive"
    if status in _RESERVE_STATUSES:
        return "reserve"
    if status in _PRACTICE_SQUAD_STATUSES:
        return "practice_squad"
    if status in _SUSPENDED_STATUSES:
        return "suspended"
    if status in _REMOVED_STATUSES:
        return "removed"
    return "other"


def build_roster_week_summaries(weekly_rosters: pd.DataFrame) -> pd.DataFrame:
    frame = weekly_rosters.copy()
    if frame.empty:
        return frame
    frame = frame.loc[frame["game_type"] == "REG"].copy()
    if frame.empty:
        return frame
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["years_exp"] = pd.to_numeric(frame.get("years_exp"), errors="coerce")
    frame["is_available"] = frame.apply(_is_available_row, axis=1)
    frame["position_bucket"] = frame["position"].map(_position_bucket)
    frame["status_bucket"] = frame["status"].map(_status_bucket)
    frame["is_rookie"] = (frame["years_exp"].fillna(0) <= 0).astype(int)
    frame["is_veteran"] = (frame["years_exp"].fillna(0) >= 5).astype(int)

    rows: list[dict[str, Any]] = []
    for (season, week, team), group in frame.groupby(["season", "week", "team"], sort=False):
        available = group.loc[group["is_available"] == 1].copy()
        inactive = group.loc[group["status_bucket"] == "inactive"].copy()
        reserve = group.loc[group["status_bucket"] == "reserve"].copy()
        suspended = group.loc[group["status_bucket"] == "suspended"].copy()
        practice = group.loc[group["status_bucket"] == "practice_squad"].copy()
        unavailable = group.loc[group["status_bucket"].isin({"inactive", "reserve", "suspended"})].copy()
        bucket_counts = available["position_bucket"].value_counts().to_dict()
        unavailable_bucket_counts = unavailable["position_bucket"].value_counts().to_dict()
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": str(team),
                "roster_size": int(len(group)),
                "available_roster_size": int(len(available)),
                "inactive_count": int(len(inactive)),
                "reserve_count": int(len(reserve)),
                "suspended_count": int(len(suspended)),
                "practice_squad_count": int(len(practice)),
                "available_qb_count": int(bucket_counts.get("qb", 0)),
                "available_skill_count": int(bucket_counts.get("skill", 0)),
                "available_offensive_line_count": int(bucket_counts.get("offensive_line", 0)),
                "available_front_seven_count": int(bucket_counts.get("front_seven", 0)),
                "available_secondary_count": int(bucket_counts.get("secondary", 0)),
                "available_specialists_count": int(bucket_counts.get("specialists", 0)),
                "unavailable_qb_count": int(unavailable_bucket_counts.get("qb", 0)),
                "unavailable_skill_count": int(unavailable_bucket_counts.get("skill", 0)),
                "unavailable_offensive_line_count": int(unavailable_bucket_counts.get("offensive_line", 0)),
                "unavailable_front_seven_count": int(unavailable_bucket_counts.get("front_seven", 0)),
                "unavailable_secondary_count": int(unavailable_bucket_counts.get("secondary", 0)),
                "available_avg_years_exp": pd.to_numeric(available["years_exp"], errors="coerce").mean(),
                "available_rookie_count": int(available["is_rookie"].sum()),
                "available_veteran_count": int(available["is_veteran"].sum()),
                "availability_score": float(min(max(len(available) / max(len(group), 1), 0.0), 1.0)),
            }
        )
    return pd.DataFrame(rows)


def attach_pregame_team_features(games: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    if games.empty or team_logs.empty:
        return games

    feature_columns = [
        column for column in team_logs.columns
        if column.endswith("_avg_last_3")
        or column.endswith("_avg_last_5")
        or column.endswith("_avg_last_10")
        or column in {"games_played_prior", "win_pct_prior", "same_site_win_pct_last_5"}
    ]
    merged = games.copy()
    for side in ("away", "home"):
        team_key_column = f"{side}_team_key"
        subset = team_logs[["game_id", "team_key", *feature_columns]].rename(
            columns={column: f"{side}_team_{column}" for column in feature_columns}
        )
        subset = subset.rename(columns={"team_key": team_key_column})
        merged = merged.merge(subset, on=["game_id", team_key_column], how="left")
    return merged


def attach_pregame_qb_features(games: pd.DataFrame, qb_logs: pd.DataFrame) -> pd.DataFrame:
    if games.empty or qb_logs.empty:
        return games
    feature_columns = [
        column for column in qb_logs.columns
        if column.endswith("_avg_last_3") or column.endswith("_avg_last_5") or column.endswith("_avg_last_10") or column == "games_played_prior"
    ]
    merged = games.copy()
    for side in ("away", "home"):
        qb_id_column = f"{side}_qb_id"
        team_column = f"{side}_team"
        subset = qb_logs[["game_id", "player_id", "team", *feature_columns]].rename(
            columns={column: f"{side}_qb_{column}" for column in feature_columns}
        )
        subset = subset.rename(columns={"player_id": qb_id_column, "team": team_column})
        merged = merged.merge(subset, on=["game_id", qb_id_column, team_column], how="left")
    return merged


def attach_pregame_roster_features(games: pd.DataFrame, roster_summaries: pd.DataFrame) -> pd.DataFrame:
    if games.empty or roster_summaries.empty:
        return games
    merged = games.copy()
    feature_columns = [column for column in roster_summaries.columns if column not in {"season", "week", "team"}]
    for side in ("away", "home"):
        team_column = f"{side}_team"
        subset = roster_summaries.rename(columns={column: f"{side}_roster_{column}" for column in feature_columns})
        subset = subset.rename(columns={"team": team_column})
        merged = merged.merge(subset, on=["season", "week", team_column], how="left")
    return merged


def add_matchup_differentials(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    paired_prefixes = (("away_team_", "home_team_", "team"), ("away_qb_", "home_qb_", "qb"), ("away_roster_", "home_roster_", "roster"))
    derived_columns: dict[str, pd.Series] = {}
    for away_prefix, home_prefix, label in paired_prefixes:
        away_columns = {column[len(away_prefix):]: column for column in frame.columns if column.startswith(away_prefix)}
        home_columns = {column[len(home_prefix):]: column for column in frame.columns if column.startswith(home_prefix)}
        for suffix in sorted(set(away_columns) & set(home_columns)):
            away_column = away_columns[suffix]
            home_column = home_columns[suffix]
            away_series = pd.to_numeric(frame[away_column], errors="coerce")
            home_series = pd.to_numeric(frame[home_column], errors="coerce")
            if away_series.notna().sum() == 0 and home_series.notna().sum() == 0:
                continue
            derived_columns[f"{label}_diff_{suffix}"] = home_series - away_series
    if not derived_columns:
        return frame
    return pd.concat([frame, pd.DataFrame(derived_columns, index=frame.index)], axis=1)
